import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns



##############################################   EXPLORE     ##############################################

def plot_variable_pairs(df):
    
    df = df.sample(n = 10000)
    
    cols = ['bedcount', 'bathcount', 'sqfeet']
    
    target = 'value'
    
    for col in cols:
        
        sns.lmplot(df, x = col, y = target, hue='county')