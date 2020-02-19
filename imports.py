import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import lightgbm.sklearn as lgb

from scipy.stats import zscore

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)

class LabelValueTransformer(TransformerMixin):
    """
    A custom transformer to label categorical columns.
    
    columns: 'auto' or a list of array-like, default='auto'
             if auto the transformer transforms all categorical
             columns.
    """
    def __init__(self, columns='auto'):

        self.columns = columns
        self.labels = {}
        
        # Custom labels
        lab_dict1 = {'Ex': 6, 'Gd': 5, 'TA': 4,
                     'Fa': 3, 'Po': 2, 'NA': 1}
        lab_dict2 = {'Y': 2, 'N': 1}
        lab_dict3 = {'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5,
                     'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1}
        lab_dict4 = {'Pave': 2,'Grvl': 1}
        lab_dict5 = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1}
        lab_dict6 = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1}
        
        self.customs = [lab_dict1, lab_dict2, lab_dict3,
                        lab_dict4, lab_dict5, lab_dict6]
        
        
    def _get_cats(self, X):
        return list(X.columns[X.dtypes == 'object'])

    
    def fit(self, X, y=None):
        # We don't want to modify the original data
        X = X.copy()
        
        # If columns not specified transform all categorical
        if self.columns == 'auto': self.columns = self._get_cats(X)

        for col in self.columns:
            
            self.labels[col] = dict()
            categories = []
            
            for i in X.index:
                val = X.loc[i, col]
                if val != np.nan:
                    if not (val in categories): categories.append(val)
            
            custom = -1
            # If the categories are those we wan't to label manually...
            for i in range(len(self.customs)):
                if set(categories) == set(self.customs[i].keys()):
                    custom = i
            
            # ...give them custom values
            if custom > -1:
                for cat in categories:
                    self.labels[col][cat] = self.customs[custom][cat]
                
            # Otherwise just give them default values
            else:
                for i in range(len(categories)):
                    self.labels[col][categories[i]] = i+1 
                 
        return self
            
        
    def transform(self, X):
        new_X = pd.DataFrame()
        
        def transform_value(val, col_name):
            if val in self.labels[col_name].keys():
                return self.labels[col_name][val]
            else: return 0

        for col in self.columns:
            new_X[col] = X[col].apply(transform_value, args=(col,))

        return new_X