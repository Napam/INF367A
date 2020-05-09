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

df, _, _ = utils.get_ml100k_data(DATA_DIR, subsample_top_users=250, subsample_top_items=250)
df[['user_id', 'item_id']] -= 1

# We are not going to use timestamp, therefore drop it
df.drop('timestamp', axis='columns', inplace=True)

def column_relabler(df: pd.DataFrame, column: str):
    uniques = pd.value_counts(df[column], sort=False).index.values
    n_uniques = len(uniques)

    # Count from 1 to conform with Stan (Stan counts indexes arrays starting at 1)
    num2id = {num_:id_ for num_, id_ in zip(range(0, n_uniques), uniques)}
    id2num = {id_:num_ for num_, id_ in zip(range(0, n_uniques), uniques)}
    
    df[column] = df[column].map(id2num)
    return id2num, num2id

df_num = df.copy()
user2num, num2user = column_relabler(df_num, 'user_id')
item2num, num2item = column_relabler(df_num, 'item_id')

# p, q represents shape of the matrix as if it was dense
p, q = len(user2num), len(item2num)

df_train, df_val = train_test_split(df_num, test_size=0.1, random_state=seed)

print(f'''Dataframe dimensions:

    df_train: {df_train.shape}
    df_val: {df_val.shape}
    ''')

models = [
    StanClasses.NormalFactorizer,
    StanClasses.NonNegativeFactorizer,
    StanClasses.ARD_Factorizer
]

init_kwargs = {'n_components':[1,2,3,4,5]}
static_kwargs = {'chains':1, 'iter':1200, 'control':{'max_treedepth':15}}

t0 = time()
hist = utils.fit_and_evaluate_models(
    models=models,
    X_train=df_train,
    X_val=df_val,
    candidate_kwargs=init_kwargs,
    static_kwargs=static_kwargs,
    ascii=True
)
evaltime = time()-t0
print('evaltime: ', evaltime)

df_hist = pd.DataFrame(hist)
df_hist.sort_values('val_mae', inplace=True)
df_hist.to_pickle('histpickle_withmodels4.pkl')

best_model = df_hist['model'].values[0]
best_params = df_hist['params'].values[0].copy()
best_params.update(static_kwargs)

df_full, _, _ = utils.get_ml100k_data(DATA_DIR)
df_full[['user_id', 'item_id']] -= 1

# We are not going to use timestamp, therefore drop it
df_full.drop('timestamp', axis='columns', inplace=True)

# final_dict4 uses 0.1 test_size, while previous ones use 0.05
df_full_train, df_full_val = train_test_split(df_full, test_size=0.1, random_state=seed)

final_model_object, fit_time, train_mae, val_mae =\
    utils.fit_and_evaluate((type(best_model), best_params, df_full_train, df_full_val))

hist2 = {
    'model':final_model_object,
    'params':best_params,
    'fit_time':fit_time,
    'train_mae':train_mae,
    'val_mae':val_mae
}

with open('final_dict4.pkl', 'wb') as f:
    pickle.dump(hist2, f)