"""Check WASS2S' installed package, classical API and bundled land geometry."""
from __future__ import annotations

import importlib
from importlib import metadata, resources
from pathlib import Path
import sys


def main():
    try:
        import wass2s
        from packaging.version import Version

        version = metadata.version("wass2s")
        if Version(version) < Version("0.4.11") or version != wass2s.__version__:
            raise RuntimeError(
                f"Versions incohérentes : distribution={version}, module={wass2s.__version__}. "
                "Réinstallez le package local avec python -m pip install -e . depuis le dossier fourni."
            )
        checks = [
            ("was_download", "WAS_Download"),
            ("was_cca", "WAS_CCA_op"),
            ("WAS_Hydro", "WAS_Hydro"),
        ]
        for module, symbol in checks:
            getattr(importlib.import_module(f"wass2s.{module}"), symbol)
        print(f"Python : {sys.executable}")
        print(f"WASS2S : {version} ; {Path(wass2s.__file__).resolve()}")
        print("API téléchargement, CCA, hydro : disponibles")
        print("Aucun téléchargement scientifique ni entraînement lancé.")
        return 0
    except (ImportError, metadata.PackageNotFoundError, RuntimeError, AttributeError) as exc:
        print(f"Environnement WASS2S incomplet : {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
