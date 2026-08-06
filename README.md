# WASS2S

> A Python framework for reproducible seasonal climate forecasting.

[![PyPI](https://img.shields.io/pypi/v/wass2s.svg)](https://pypi.org/project/wass2s/)
[![Documentation](https://readthedocs.org/projects/wass2s/badge/)](https://wass2s.readthedocs.io)
[![License](https://img.shields.io/github/license/hmandela/WASS2S)](LICENSE)

**WASS2S** is an open-source Python framework for developing, verifying, and producing seasonal climate forecasts. Designed for operational climate services, researchers, and National Meteorological and Hydrological Services (NMHSs), it provides a reproducible workflow covering data acquisition, preprocessing, model development, verification, and forecast generation.

The framework follows the recommendations of the **World Meteorological Organization (WMO)** for objective and reproducible seasonal forecasting while supporting both traditional statistical methods and modern machine learning approaches.

---

## Features

- End-to-end seasonal forecasting workflow.
- Statistical and machine learning forecasting methods.
- Model verification using deterministic and probabilistic skill metrics.
- Automated download of seasonal forecast datasets.
- Interactive Jupyter notebooks.
- Publication-quality maps and graphics.
- Reproducible environments powered by Pixi.
- Modular architecture for extending forecasting methods.

---

## Installation

### Recommended: Pixi

WASS2S is developed with **Pixi**, which automatically manages all Python, geospatial, and machine learning dependencies.

Install Pixi by following the official guide:

https://pixi.sh/latest/installation/

Clone the repository:

```bash
git clone https://github.com/hmandela/WASS2S.git
cd WASS2S
```

Start the project environment:

```bash
pixi shell
```

Pixi automatically creates the environment the first time it is used.

You can also execute commands without activating the shell:

```bash
pixi run python
pixi run jupyter lab
```

### Activate WASS2S from any directory

Instead of changing into the project directory each time, you can start the environment directly:

```bash
pixi shell --manifest-path /path/to/WASS2S/pyproject.toml
```

#### Bash (Linux/macOS)

Add the following function to your `~/.bashrc` or `~/.zshrc`.

Replace `/path/to/WASS2S` with the location of your local repository.

```bash
wass2s() {
    eval "$(pixi shell-hook \
        --shell bash \
        --manifest-path /path/to/WASS2S/pyproject.toml)"
}
```

Reload your shell:

```bash
source ~/.bashrc
```

You can now activate WASS2S from anywhere:

```bash
wass2s
```

#### PowerShell (Windows)

Open your PowerShell profile:

```powershell
if (!(Test-Path $PROFILE)) {
    New-Item -ItemType File -Path $PROFILE -Force
}
notepad $PROFILE
```

Add:

```powershell
function wass2s {
    pixi shell --manifest-path "C:\path\to\WASS2S\pyproject.toml"
}
```

Replace `C:\path\to\WASS2S` with your local repository.

Reload your profile:

```powershell
. $PROFILE
```

Then simply run

```powershell
wass2s
```

---

### Install from PyPI

If you only need the Python package,

```bash
pip install wass2s
```

---

### Legacy Conda Environment

Legacy Conda environments remain available:

```bash
conda env create -f WAS_S2S_linux.yml
conda activate WASS2S
```

or

```bash
conda env create -f WAS_S2S_windows.yml
conda activate WASS2S
```

---

## Tutorial Notebooks

Example notebooks are available in the companion repository:

```bash
git clone https://github.com/hmandela/WASS2S_notebooks.git
```

---

## Climate Data Access

WASS2S supports downloading seasonal forecast datasets from the **Copernicus Climate Data Store (CDS)** and other supported data providers.

To enable downloads, create a CDS account and configure your API credentials as described [here](https://cds.climate.copernicus.eu/how-to-api).

---

## Documentation

Comprehensive documentation is available online.

- User Guide
- Installation
- Tutorials
- API Reference
- Training Material

Documentation:

- https://wass2s.readthedocs.io
- https://hmandela.github.io/WAS_S2S_Training/

---

## Support

- Questions: GitHub Discussions
- Bug reports: GitHub Issues
- Documentation: Read the User Guide

---

## Contributing

Contributions are welcome.

Whether you're fixing bugs, improving documentation, implementing new forecasting methods, or adding tests, your contributions help improve WASS2S for the entire community.

Please read the [Contributing Guide](CONTRIBUTING.md) before opening an issue or submitting a pull request.

---

## Code of Conduct

To foster an open and welcoming community, all contributors are expected to follow the project's [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Citation

If WASS2S contributes to your research, please cite the software.

GitHub provides a citation through the repository's **Cite this repository** button. Citation metadata are also available in `CITATION.cff`.

---

## Acknowledgments

WASS2S has been developed with support from the **Accelerating Impacts of CGIAR Climate Research for Africa (AICCRA)** project and the **AGRHYMET Regional Climate Centre for West Africa and the Sahel (AGRHYMET RCC-WAS)**.

We thank the participants of the *Training on the New Generation of Seasonal Forecasting in West Africa and the Sahel* for their valuable feedback and contributions.

WASS2S builds upon numerous open-source scientific software projects, including **xarray**, **scikit-learn**, **xeofs**, **xcast**, **xskillscore**, **Cartopy**, **NumPy**, **SciPy**, **Matplotlib**, and many others. We gratefully acknowledge their developers and maintainers.

---

## License

WASS2S is distributed under the **GNU General Public License v3.0 (GPL-3.0)**.

See the [LICENSE](LICENSE) file for details.