import pandas as pd
import numpy as np
from math import pi

def generate_debug_time_series(period=60):

    # Création de l'index de la série temporelle
    dates = pd.date_range(start='2020-01-01', 
                          end='2024-05-01', 
                          freq='B', # 'B' pour les jours ouvrés
                          name='ds')  
    
    # Création des données (prix augmentant de 1 chaque jour ouvré)
    debug_series = pd.DataFrame({'y': [f(x, period) for x in range(len(dates))]}, index = dates)
    
    return debug_series

def f(x, period):
    return (np.cos(2*pi*x/period) + 1)/2

if (__name__ == '__main__'):
    debug_series = generate_debug_time_series()
    print(debug_series)