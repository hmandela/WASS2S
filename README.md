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

4. Download notebooks for simulation
```bash
git clone https://github.com/hmandela/WASS2S_notebooks.git
```
5. Create CDS API key and use it to download NMME and C3S models data from the Climate Data Store (CDS) and IRI Data Library.

-   Create an account with Copernicus by signing up [here](https://cds.climate.copernicus.eu/datasets)

-   Once you successfully create an account, kindly log in to your Copernicus account and click on your name at the top right corner of the page. Note your "UID" and "Personal Access Token key". 


-  Configure .cdsapirc file.

In your activated terminal, kindly initiate the Python interpreter by entering the command python3. Subsequently, carefully copy and paste the below code, ensuring to replace "Personal Access Token" with yours.

```python
import os

config_data = '''url: https://cds.climate.copernicus.eu/api
key: Personal Access Token
verify: 0
'''

path_to_home = "/".join([os.path.expanduser('~'),".cdsapirc"])

if not os.path.exists(path_to_home):
    with open(path_to_home, 'w') as file:
        file.write(config_data)
        
print("Configuration file created successfully!")
```
### Upgrade wass2s
If you want to upgrade wass2s to a newer version, use the following command:
```bash
pip install --upgrade wass2s
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
I would like to express my sincere gratitude to all the participants of the **job-training on the new generation of seasonal forecasts in West Africa and the Sahel**.  
Your valuable feedback has significantly contributed to the improvement of this tool. I look forward to continuing to receive your insights and, where possible, your contributions.  
**A seed has been planted within you—now, let’s grow it together.**

We also extend our heartfelt thanks to the **AICCRA project** for supporting this development, and to **Dr. Abdou ALI**, Head of the **Climate-Water-Meteorology Department at AGRHYMET RCC-WAS**, for his guidance and support.
---

📖 For more detailed information, tutorials, and support, please visit the [WAS_S2S Training Documentation](https://hmandela.github.io/WAS_S2S_Training/).

