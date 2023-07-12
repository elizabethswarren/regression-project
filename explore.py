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


def plot_categorical_and_continuous_vars(df, cat_var_col, con_var_col):
    
    df = df.sample(n=1000)
    
    fig, axs = plt.subplots(1,3, figsize=(18,8))
    
    sns.stripplot(ax=axs[0], x=cat_var_col, y=con_var_col, data=df)
    axs[0].set_title('stripplot')
    
    sns.boxplot(ax=axs[1], x=cat_var_col, y=con_var_col, data=df)
    axs[1].set_title('boxplot')
    
    sns.swarmplot(ax=axs[2], x=cat_var_col, y=con_var_col, data=df, s=1)
    axs[2].set_title('swarmplot')