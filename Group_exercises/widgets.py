# import numpy as np 
# import pandas as pd 
# from calculator import f 

p_unreliable = 0.7

def likelihood(data: list, theta: int):
    cum = 1 
    n = len(data)
    n_u_cum = 0
    for i, x in enumerate(data):
        # print(f'{cum:.4f}')
        if x == 'r':
            cum *= 1 - ((theta-n_u_cum) / (n-i))
        if x == 'u':
            cum *= ((theta-n_u_cum) / (n-i))
            n_u_cum += 1
    return cum

def posterior(data: list, theta: int):
    n = len(data)
    lhood = likelihood(data, theta) * p_unreliable**theta
    p_data = 0
    for i in range(n):
        p_data += likelihood(data, i) * p_unreliable**i
    return lhood / p_data

if __name__ == '__main__':
    D = list('rrrrrrrrrrrurrrrrrrr')
    # l = likelihood(D, 1)
    post = posterior(D, 1)
    print(post)
    



