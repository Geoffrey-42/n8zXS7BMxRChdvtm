from setuptools import setup, find_packages

setup(
    name='Trading_Recommender',
    version='1.0.0',
    author='Geoffrey Bonias',
    author_email='geoffreybonias1@gmail.com',
    description='Stock Price Forecaster and Recommender Simulation System',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.4',
        'matplotlib>=3.8.0',
        'yfinance>=0.2.38',
        'pandas>=2.1.4',
        'seaborn>=0.12.2',
        'prophet>=1.1.5',
        'statsmodels>=0.14.0',
        'pmdarima>=2.0.4',
        'tensorflow>=2.16.1',
        'scikit-learn>=1.2.2',
        'optuna>=3.6.1',
        'optuna-dashboard>=0.15.1'
    ],
)

