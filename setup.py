from setuptools import setup, find_packages
import os

setup(
    name="wass2s",
    version="0.2.19",
    author="HOUNGNIBO C. M. Mandela",
    author_email="hmandelahmadiba@gmail.com",
    description="A Python package for seasonal climate forecast.",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/hmandela/WASS2S",
    packages=find_packages(),
    install_requires=[
        "statsmodels",
        "pykrige",
        "cartopy",
        "scipy==1.11.3",
        "matplotlib==3.7.3",
        "xeofs==3.0.4",
        "xskillscore==0.0.26",
        "properscoring",
        "cdsapi==0.7.4",
        "rasterio",
        "geopandas",
        "shapely",
        "rioxarray"
    ],
    python_requires=">=3.12",
    license="GPL-3.0",
    license_files=('LICENSE',),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
