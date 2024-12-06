import pandas as pd
import numpy as np
import random
import os
from collections import deque
from generator import GameState, ECellType, getRandomMove, GetNextMove
from generator import SerializeMoveAsString_cyclic1

def generate_unique(method_func, size):
    STATES_NUMBER =  round(1.1 * size + 1000) # so that beginning states had no overbalance 
    dct = {'before': [], 'move': [], 'after': []}
    Q = deque([(GameState(), ECellType.First)])
    for _ in range(STATES_NUMBER):
        state, cellType = Q.popleft()
        if state.IsFinish():
            Q.appendleft((GameState(), ECellType.First))
        else:
            move = getRandomMove(state, cellType)
            
            dct['before'].append(state.SerializeAsString())
            dct['move'].append(method_func(move))
            state.ApplyMove(move)
            dct['after'].append(state.SerializeAsString())
    
            Q.append((state, GetNextMove(cellType)))
        
    df = pd.DataFrame.from_dict(dct).drop_duplicates(subset=["before", "move"])
    data = df.sample(frac=1)[:size]
    assert data.shape[0] == size
    return data


def get_datasets(method_func, k, path):
    test_size = k
    val_size = k
    train_size = 2*k
    df = generate_unique(method_func, size=4*k)
    r = np.array(train_size * ['train'] + val_size * ['val'] + test_size * ['test'])
    np.random.shuffle(r)
    train = df[r == 'train'][['before', 'move', 'after']].sample(frac=1)
    val = df[r == 'val'][['before', 'move', 'after']].sample(frac=1)
    test = df[r == 'test'][['before', 'move', 'after']].sample(frac=1)
    
    try:
        os.mkdir(f"{path}/{k}")
    except OSError:
        pass
    
    train.to_csv(f"{path}/{k}/train_{train_size}.csv", index=False, header=False)
    val.to_csv(f"{path}/{k}/val_{val_size}.csv", index=False, header=False)
    test.to_csv(f"{path}/{k}/test_{test_size}.csv", index=False, header=False)
    return train, val, test

if __name__ == "__main__":
    # np.random.seed(42)
    # random.seed(42)
    path = "data/cyclic1"
    try:
        os.mkdir(path)
    except OSError as error:
        pass
    
    train, test, val = get_datasets(method_func=SerializeMoveAsString_cyclic1, k=11, path=path)