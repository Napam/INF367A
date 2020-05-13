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
from typing import Union, Iterable
from sklearn.model_selection import ParameterGrid
from multiprocessing import Pool
from tqdm import tqdm
from time import time

def fit_and_evaluate(args: tuple):
    '''
    Fits and evaluate model from StanClasses.py

    Parameter
    ---------
    args: tuple as (model class, init_kwargs, X_train, X_val) or
          (integer index, model class, init_kwargs, X_train, X_val)

          This function is used for parallell processing, therefore the weird 
          argument setup. The integer index is used to keep track of order 
          when using parallell processing. 

    Returns
    -------
    (model_object, fit_time, train_mae, val_mae) or
    (integer index, model_object, fit_time, train_mae, val_mae)
    '''
    with_index = False
    try:
        model, init_kwargs, X_train, X_val = args
    except ValueError:
        index, model, init_kwargs, X_train, X_val = args
        with_index = True

    model_object = model(**init_kwargs)
 
    t0 = time()
    model_object.fit(X_train)
    fit_time = time()-t0
    
    train_mae = model_object.mae(X_train)
    
    if X_val is not None:
        val_mae = model_object.mae(X_val)   
    else:
        val_mae = None
    
    if with_index:
        return index, model_object, fit_time, train_mae, val_mae
    else:
        return model_object, fit_time, train_mae, val_mae

def fit_and_evaluate_models(models: Iterable, X_train: Union[np.ndarray, pd.DataFrame], 
                            X_val: Union[np.ndarray, pd.DataFrame]=None, 
                            candidate_kwargs: dict={}, static_kwargs: dict={}, 
                            verbose=True, ascii=False):
    '''
    Given models, and candidate kwargs, trains the models on every combination candidate_kwargs.

    Parameters
    ----------
    models: Iterable of model classes

    X_train: Dataset used for training

    X_val: Dataset used for validation

    candidate_kwargs: Keyword arguemts to be trained on. They will be turned into a 
                      grid of all combinations. Think of how GridSearch works. 

    static_kwargs: Shared keyword arguments for all the models. 

    verbose: Shows tqdm if True, True by default

    ascii: Whether tqdm bar should print ascii (for compatibility)

    Returns
    -------
    hist: dictionary containing fitted model object, the corresponding parameters,
          fit_time, train_mae, val_mae.
    '''

    hist = {'model':[], 'params':[], 'fit_time':[], 'train_mae':[], 'val_mae':[]}
    
    map_args = []
    candidate_param_list = []
    param_gen = ParameterGrid({'model':models, **candidate_kwargs})
    n_params = len(param_gen)

    for i, paramdict in enumerate(param_gen):
        model = paramdict.pop('model')
        
        candidate_param_list.append(paramdict)
        
        paramdict = paramdict.copy()
        paramdict.update(static_kwargs)
        
        map_args.append((i, model, paramdict, X_train, X_val))
    
    # Train candidates on separate processes 
    with Pool(None) as p:
        fit_iterator = tqdm(
            p.imap_unordered(fit_and_evaluate, map_args), total=n_params,
            desc='Fitting models', disable=not verbose, unit='model', position=0,
            ascii=ascii
        )
        # Note that imap_unordered is used, so the ordering will probably be 
        # scrambled. 
        results = list(fit_iterator)
        
    for result in results:
        index, model_object, fit_time, train_mae, val_mae = result
        hist['model'].append(model_object)
        hist['params'].append(candidate_param_list[index])
        hist['fit_time'].append(fit_time)
        hist['train_mae'].append(train_mae)
        hist['val_mae'].append(val_mae)
    
    return hist

def get_stan_code(filename: str):
    '''
    Given filename, read the text and return it. 
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
        return str.join(' ', lines)

def StanModel_cache(model_code, model_name=None, cache_dir='stan_cache', **kwargs):
    """
    Use just as you would `stan`
    
    Copy pasted from Stan documentation. This function caches compiled Stan 
    models.

    I have added some text cleaning so it wont recompile if I add comments or 
    newlines or whatever. 
    """
    
    # Clean text to avoid stupid recompilations

    # Removes multiline comments
    to_hash = re.sub('(\/\*(.|\n)*?\*\/)', '', model_code)
    # Removes comments
    to_hash = re.sub('(//.+)', '', to_hash)
    # Removes trailing spaces
    to_hash = re.sub('([ \t]{2,})', ' ', to_hash)
    # Removes empty space
    to_hash = re.sub('(\s+\n)', '\n', to_hash)
    # Removes repeated newlines
    to_hash = re.sub('(\n{2,})', '\n', to_hash)

    code_hash = md5(to_hash.encode('ascii')).hexdigest()
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

    Parameters
    ----------
    directory: path to directory with the ml100k files
    
    subsample_top_users: If given an integer, it subsamples the dataset such that
                         you get the top users based on number of movies they have 
                         rated.

    subsample_top_items: If given an integer, it subsamples dataset such that you 
                         get the top movies based on the number reviews they have
                         gotten. If subsample_top_users is specified as well, it 
                         will choose the top movies based on said subsampled users. 

    Returns
    --------
    (df_users, df_item_features, df_item_metadata)
    df_users is the data "matrix" in sparse format. 
    df_item_features is the genre infromation for the movies. Not really used. 
    df_item_metadata I dont really remember what it was. I dont really use it. 
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
    df_users, _, _ = get_ml100k_data(directory='ml-100k')

    # Start counting at zero instead of one
    df_users[['user_id', 'item_id']] -= 1 

    # Pick out some random users to make test set 


    