import numpy as np
import random

def shuffle(order,min_idx, max_idx):
    # Get idx to swap
    swap_idx = random.sample(range(min_idx, max_idx), 2)
    # Swap
    temp1 = order[swap_idx[0]]
    order[swap_idx[0]] = order[swap_idx[1]]
    order[swap_idx[1]] = temp1

def shuffle_N_times(order, N):
    order_copy = order.copy()
    for i in range(N):
        shuffle(order_copy,0,len(order))
        
    return order_copy

def shuffle_first_i(order, N, i):
    order_copy = order.copy()
    
    for j in range(N):
        shuffle(order_copy,0,i)
        
    return order_copy

def shuffle_last_i(order, N, i):
    order_copy = order.copy()
    
    for j in range(N):
        shuffle(order_copy,len(order)-i,len(order))
        
    return order_copy