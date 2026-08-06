# Contributing to WASS2S

Thank you for your interest in contributing to **WASS2S**! We welcome contributions of all kinds, including bug reports, feature requests, documentation improvements, new forecasting methods, and code contributions.

If you're new to open source, don't worry—we're happy to help you get started.

## Fixing typos

Small improvements to the documentation, such as fixing typos, grammar, or clarifying wording, can be made directly through GitHub's web interface.

When editing documentation, please update the source files (Markdown, docstrings, or notebooks) rather than generated documentation.

## Bigger changes

For larger changes, we recommend opening an issue first to discuss your proposal before starting development. This helps ensure that your work aligns with the project's goals and avoids duplicated effort.

If you've found a bug, please open an issue and include:

- a clear description of the problem;
- steps to reproduce the issue;
- a minimal reproducible example, if possible;
- your operating system, Python version, and WASS2S version.

## Pull Request Process

1. Fork the repository and clone it locally.

   ```bash
   git clone https://github.com/<your-username>/WASS2S.git
   cd WASS2S
   ```

2. Install the development environment.

   ```bash
   pixi install
   pixi shell
   ```

3. Create a branch for your work.

   ```bash
   git checkout -b feature/brief-description
   ```

4. Make your changes.

5. Format and lint the code.

   ```bash
   black .
   ruff check .
   ```

6. Run the test suite.

   ```bash
   pytest
   ```

7. Commit your changes and push your branch.

8. Open a Pull Request.

Please give your Pull Request a concise title and include a description of the changes. If your Pull Request addresses an existing issue, include

```
Fixes #<issue-number>
```

in the description.

For user-facing changes, please add a brief entry to the **Unreleased** section of `CHANGELOG.md` (or `NEWS.md`, if used).

## Code Style

New code should follow the project's coding conventions.

- Follow **PEP 8**.
- Format code with **Black**.
- Use **Ruff** for linting.
- Write clear docstrings using the NumPy docstring style.
- Keep functions focused and modular.
- Add type hints where appropriate.

Please avoid unrelated code formatting changes in the same Pull Request.

## Testing

We use **pytest** for unit testing.

New functionality should include appropriate tests whenever possible. Bug fixes should include a regression test when practical.

A Pull Request should pass the full test suite before it is reviewed.

## Documentation

Documentation is an important part of WASS2S.

If your contribution changes user-facing functionality, please update the relevant documentation, tutorials, notebooks, or examples.

## Questions

If you have questions about contributing or using WASS2S, feel free to open a GitHub Discussion before opening an issue.

## Code of Conduct

By participating in this project, you agree to abide by the project's [Code of Conduct](CODE_OF_CONDUCT.md).