import numpy as np
from typing import Union
# import sys

# np.set_printoptions(formatter={'float_kind': lambda x: f'{x:.2f}'})
# np.set_printoptions(threshold=np.inf)

def normalize_columns(X: np.ndarray):
    norms = np.linalg.norm(X, axis=0)   
    return np.nan_to_num((X.T / norms.reshape(-1,1)).T)

def texmatrix(a: np.ndarray, convert_type: Union[bool, type] = False, mtype: str = 'pmatrix') -> str:
    '''Converts numpy array to latex matrix'''
    if convert_type:
        a = a.astype(convert_type)

    lines = str(a).replace('[', '').replace(']', '').splitlines()
    tex = [r'\begin{{{}}}'.format(mtype)]
    tex += ['  ' + ' & '.join(l.split()) + r' \\' for l in lines]
    tex +=  [r'\end{{{}}}'.format(mtype)]
    return '\n'.join(tex)


def npmatrix(a: str) -> np.ndarray:
    '''Converts tex matrix into numpy array'''
    tex = '[[' + a.replace('\\', '],[').replace('&', ',').replace('\n', '') + ']]'
    return np.array(eval(tex))
    

if __name__ == '__main__':
    test =\
    '''
  4 & 0 & 0 \\
        0 & 0 & 0 \\
        0 & 0 & 0 
    '''

    print(npmatrix(test))

