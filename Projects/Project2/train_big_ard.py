import numpy as np
seed = 42069
np.random.seed(seed)
import pandas as pd
from matplotlib import pyplot as plt
import arviz
import pystan
from scipy import sparse, stats
from typing import Iterable, Union, Callable
from sklearn.model_selection import train_test_split, ParameterGrid
import altair as alt
from time import time, sleep
from tqdm import tqdm
from multiprocessing import Pool
import pickle

# Own files
import utils 
import StanClasses
    

# Define constants
DATA_DIR = 'ml-100k'

df, _, _ = utils.get_ml100k_data(DATA_DIR)
df[['user_id', 'item_id']] -= 1

# We are not going to use timestamp, therefore drop it
df.drop('timestamp', axis='columns', inplace=True)

df_train, df_val = train_test_split(df, test_size=0.1, random_state=seed)

print(f'''Dataframe dimensions:

    df_train: {df_train.shape}
    df_val: {df_val.shape}
    ''')

init_kwargs = {'n_components':3}
static_kwargs = {'chains':1, 'iter':2000, 'control':{'max_treedepth':15}}

init_kwargs.update(static_kwargs)

df_train, df_val = train_test_split(df, test_size=0.1, random_state=seed)

model_object = StanClasses.ARD_Factorizer(**init_kwargs)

fit_time = time()
model_object.fit(df_train)
fit_time = time() - fit_time

hist = {
    'model':model_object,
    'params':init_kwargs,
    'fit_time':fit_time,
}

with open('big_dict_ard.pkl', 'wb') as f:
    pickle.dump(hist, f)