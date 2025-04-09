# wass2s: A python-based tool for seasonal climate forecast

![Modules in WAS_S2S](./modules.png)

**wass2s** is a comprehensive tool developed to enhance the accuracy and reproducibility of seasonal forecasts in West Africa and the Sahel. This initiative aligns with the World Meteorological Organization's (WMO) guidelines for objective, operational, and scientifically rigorous seasonal forecasting methods.


## Overview
The wass2s tool is designed to facilitate the generation of seasonal forecasts using various statistical and machine learning methods including the Exploration of AI methods. 
It helps forecaster to download data, build models, verify the models, and forecast. A user-friendly jupyter-lab notebook streaming the process of prevision.

## 🚀 Features

- ✅ **Automated Forecasting**: Streamlines the seasonal forecasting process, reducing manual interventions.
- 🔄 **Reproducibility**: Ensures that forecasts can be consistently reproduced and evaluated.
- 📊 **Modularity**: Highly modular tool. Users can easily customize and extend the tool to meet their specific needs.
- 🤖 **Exploration of AI and Machine Learning**: Investigates the use of advanced technologies to further improve forecasting accuracy.

## 📥 Installation
1.  Download and Install miniconda

-   For Windows, download the executable [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)

-   For Linux (Ubuntu), in the terminal run:

    ``` bash
    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install wget
    wget -c -r https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --no-check-certificate
    bash Miniconda3-latest-Linux-x86_64.sh
    ```
2. Create an environment and activate
- For Windows: download yaml [here](https://github.com/hmandela/WASS2S/blob/main/WAS_S2S_windows.yml) and run
```bash
conda env create -f WAS_S2S_windows.yml
conda activate WASS2S
```
- For Linux: download yaml [here](https://github.com/hmandela/WASS2S/blob/main/WAS_S2S_linux.yml) and run
```bash
conda env create -f WAS_S2S_linux.yml
conda activate WASS2S
```

3. Install wass2s
```bash
pip install wass2s
```

## ⚙️ Usage

Comprehensive usage guidelines, including data preparation, model configuration, and execution steps, are available in the [WAS_S2S Training Documentation](https://hmandela.github.io/WAS_S2S_Training/).

## 🤝 Contributing

We welcome contributions from the community to enhance the `WAS_S2S` tool. Please refer to our [contribution guidelines](CONTRIBUTING.md) for more information.

## 📜 License

This project is licensed under the [GPL-3 License](https://github.com/hmandela/WASS2S/blob/main/LICENSE.txt).

## Contact

For questions or support, please open a [Github issue](https://github.com/hmandela/WAS_S2S/issues).

## Credits

- scikit-learn: [scikit-learn](https://scikit-learn.org/stable/)
- EOF analysis: [xeofs](https://github.com/xarray-contrib/xeofs/tree/main) 
- xcast: [xcast](https://github.com/kjhall01/xcast/)
- xskillscore: [xskillscore](https://github.com/xarray-contrib/xskillscore)
- ... and many more!

## 🙌 Acknowledgments
I would like to express my gratitude to the participants of the action-training on the new generation of seasonal forecasts in West Africa and the Sahel. Your feedback has greatly contributed to the improvement of this tool. I hope to continue receiving your insights and, if possible, your contributions. A seed has been planted within you… now, let’s grow it together.

We extend our gratitude to the AICCRA project for supporting the development and to Dr Abdou ALI Head of AGRHYMET RCC-WAS Climate-Water-Meteorology Department.
---

📖 For more detailed information, tutorials, and support, please visit the [WAS_S2S Training Documentation](https://hmandela.github.io/WAS_S2S_Training/).

