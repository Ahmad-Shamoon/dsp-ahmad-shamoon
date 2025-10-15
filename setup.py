from setuptools import setup, find_packages

setup(
    name="house_prices",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib"
    ],
    author="Ahmad Shamoon",
    description="A simple package for training and predicting house prices.",
)
