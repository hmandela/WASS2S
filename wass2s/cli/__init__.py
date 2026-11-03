"""Operational seasonal forecast workflows for precipitation and temperature."""


def run_forecast(variable, argv=None):
    """Run the PRCP or TEMP pipeline with a sequence of command-line arguments.

    If argv is omitted, use the arguments of the current process.
    In a notebook, pass a list of arguments, or [] for defaults.
    """
    from .entrypoints import main

    return main(variable, argv)
