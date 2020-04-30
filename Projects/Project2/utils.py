import numpy as np
seed = 42069
np.random.seed(seed)
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import sparse
import re
import os
from io import StringIO
from abc import abstractmethod
from sklearn.model_selection import train_test_split
import pystan
import pickle
from hashlib import md5
from typing import Union

def histmax(chain: np.ndarray, bins: Union[str, int]='auto'):
    hist, edges = np.histogram(chain, bins=bins)
    hat = edges[hist.argmax()+1]
    return hat

def get_stan_code(filename: str):
    with open(filename, 'r') as f:
        lines = f.readlines()
        return str.join(' ', lines)

def StanModel_cache(model_code, model_name=None, cache_dir='stan_cache', **kwargs):
    """Use just as you would `stan`"""
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)

    path = os.path.join(cache_dir, cache_fn)

    try:
        sm = pickle.load(open(path, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code, **kwargs)
        with open(path, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm

def get_ml100k_data(directory:str, subsample_top_users: int=None, 
                    subsample_top_items: int=None):
    '''
    Gets ml-100k data
    '''
    data_path = os.path.join(directory, 'u.data')
    genre_path = os.path.join(directory, 'u.genre')
    item_path = os.path.join(directory, 'u.item')

    df_users = pd.read_csv(data_path,
                           delim_whitespace=True,
                           names=['user_id', 'item_id', 'rating', 'timestamp'])

    df_genres = pd.read_csv(genre_path, delimiter='|', header=None,
                            names=['genre','genre_id'])

    item_features = ['movie_id','title','release_date','url'] +\
                     list(df_genres.genre.values)

    with open(item_path, 'r') as f:
        fixed_item_data = f.read().replace('|http','http')
        item_file = StringIO(fixed_item_data)

    df_items = pd.read_csv(item_file, delimiter='|', names=item_features)

    df_item_metadata = df_items.iloc[:,:4]
    df_item_features = df_items.iloc[:,4:]

    if subsample_top_users is not None:
        # pd.value_counts sorts by default
        topusers = pd.value_counts(df_users.user_id).index.values[:subsample_top_users]
        df_users = df_users.loc[df_users.user_id.isin(topusers)]

    if subsample_top_items is not None:
        # pd.value_counts sorts by default
        topitems = pd.value_counts(df_users.item_id).index.values[:subsample_top_items]
        df_users = df_users.loc[df_users.item_id.isin(topitems)]

    return df_users, df_item_features.astype(float), df_item_metadata

if __name__ == '__main__':
    # df_users, _, _ = get_ml100k_data(directory='ml-100k')

    # Start counting at zero instead of one
    # df_users[['user_id', 'item_id']] -= 1 

    # Pick out some random users to make test set 

    # df_train, df_test = train_test_split(df_users, test_size=0.3)

    # X_train = sparse.csc_matrix((df_train.rating, (df_train.user_id, df_train.item_id))).toarray()
    # X_test = sparse.csc_matrix((df_test.rating, (df_test.user_id, df_test.item_id))).toarray()

    # print(df_users)
    # print(X_train)
    # print(X_test)

    A = np.random.randint(0,100, size=(10,8))
    from sklearn.decomposition import PCA, NMF
    P = NMF(2)
    X = P.fit_transform(A)
    print(A)
    print(X.shape)
    print(X)
    print(P.components_.shape)
    print(P.components_)
    print((X@P.components_).astype(int))