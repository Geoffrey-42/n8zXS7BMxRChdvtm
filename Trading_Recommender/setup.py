from setuptools import setup, find_packages

setup(
    name='Trading_Recommender',
    version='1.0.0',
    author='Geoffrey Bonias',
    author_email='geoffreybonias1@gmail.com',
    description='Stock Price Forecaster & Decision Support System Simulator',
    packages=find_packages(),
    python_requires='==3.11.7'
    install_requires=[
        'numpy==1.26.4',
        'matplotlib==3.8.0',
        'yfinance==0.2.38',
        'pandas==2.1.4',
        'tsfracdiff==1.0.4,
        'ta==0.11.0',
        'pandas-ta==0.3.14',
        'seaborn==0.12.2',
        'prophet==1.1.5',
        'scikit-learn==1.2.2',
        'tslearn==0.6.3',
        'sktim==0.29.0',
        'statsmodels==0.14.0',
        'pmdarima==2.0.4',
        'tensorflow==2.16.1',
        'optuna==3.6.1',
        'optuna-dashboard==0.15.1'
    ],
)

# Note: to install pygooglenews, you may need to downgrade setuptools first:
    # pip install "setuptools<58.0.0"
    # pip install django-celery
    # pip install pygooglenews
    # Successfully installed dateparser-0.7.6 feedparser-5.2.1 pygooglenews-0.1.2
    # pip install setuptools --upgrade
    # Note: setuptools version used == 69.5.1
    # pip install feedparser --upgrade
    # Note: feedparser version used == 6.0.11

