"""Custom quartodoc renderer for wass2s.

quartodoc 0.11.1's MdRenderer only implements rendering for a subset of the
docstring section kinds griffe can produce. wass2s docstrings use several
valid, standard numpydoc sections — "Methods" / "Classes" (module- and
class-level indexes) and "Yields" / "Receives" (generator functions) — that
griffe parses into DocstringSectionFunctions / DocstringSectionClasses /
DocstringSectionYields / DocstringSectionReceives objects. MdRenderer has no
`render()` overload for these, so it raises `NotImplementedError` and aborts
the whole build.

This subclass adds rendering for those section kinds so the build completes
without needing to rewrite the source docstrings.
"""

from typing import Union

from quartodoc._griffe_compat import docstrings as ds
from quartodoc.renderers.md_renderer import MdRenderer, ParamRow
from plum import dispatch


class Renderer(MdRenderer):
    style = "wass2s_markdown"

    @dispatch
    def render(self, el: ds.DocstringSectionFunctions):
        items = [f"* **{item.name}**: {item.description}" for item in el.value]
        return "\n".join(items)

    @dispatch
    def render(self, el: ds.DocstringSectionClasses):
        items = [f"* **{item.name}**: {item.description}" for item in el.value]
        return "\n".join(items)

    @dispatch
    def render(self, el: Union[ds.DocstringSectionYields, ds.DocstringSectionReceives]):
        rows = list(map(self.render, el.value))
        header = ["Name", "Type", "Description"]
        return self._render_table(rows, header, "returns")

    @dispatch
    def render(self, el: Union[ds.DocstringYield, ds.DocstringReceive]) -> ParamRow:
        return ParamRow(
            el.name,
            el.description,
            annotation=self.render_annotation(el.annotation),
        )
