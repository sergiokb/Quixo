import pandas as pd
import numpy as np
import random
import os
from collections import deque
from generator import GameState, ECellType, getRandomMove, GetNextMove
from generator import SerializeMoveAsString_v2, SerializeMoveAsString_v4
from generator import SerializeMoveAsString_cyclic1
from generator import SerializeMoveAsString_linear1


def generate_unique(size, method_func, process=None):
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
            if process is None:
                state.ApplyMove(move)
                after = state.SerializeAsString()
            elif process == "reasoning":
                after = state.SerializeAsStringOmmiting(move.row, move.column)
                state.ApplyMove(move)
                after += "\n"
                after += state.SerializeAsStringOmmiting(*move.NewPosition())
                after += "\n"
                after += state.SerializeAsString()
            dct['after'].append(after)
    
            Q.append((state, GetNextMove(cellType)))
        
    df = pd.DataFrame.from_dict(dct).drop_duplicates(subset=["before", "move"])
    data = df.sample(frac=1)[:size]
    assert data.shape[0] == size
    return data

def get_datasets(path, train_sizes, val_size, test_size, method_func, process=None):
    full_size = sum(train_sizes) + val_size + test_size
    df = generate_unique(full_size, method_func, process)
    
    r = []
    for i, train_size in enumerate(train_sizes):
        r += train_size * [f'{i}_train_{train_size}']
    r += test_size * [f'test_{test_size}']
    r += val_size * [f'val_{val_size}']
    
    r = np.array(r)
    np.random.shuffle(r)
    
    
    for i, train_size in enumerate(train_sizes):
        train = df[r == f'{i}_train_{train_size}'][['before', 'move', 'after']].sample(frac=1)
        assert len(train) == train_size
        train.to_csv(f"{path}/{i}_train_{train_size}.csv", index=False)
        
    val = df[r == f'val_{val_size}'][['before', 'move', 'after']].sample(frac=1)
    assert len(val) == val_size
    val.to_csv(f"{path}/val_{val_size}.csv", index=False)
    
    test = df[r == f'test_{test_size}'][['before', 'move', 'after']].sample(frac=1)
    assert len(test) == test_size
    test.to_csv(f"{path}/test_{test_size}.csv", index=False)
    
    return train, val, test

# def get_datasets(path, k, method_func, process=None):
#     test_size = k
#     val_size = k
#     train_size = 2*k
#     df = generate_unique(4*k, method_func, process)
#     r = np.array(train_size * ['train'] + val_size * ['val'] + test_size * ['test'])
#     np.random.shuffle(r)
#     train = df[r == 'train'][['before', 'move', 'after']].sample(frac=1)
#     val = df[r == 'val'][['before', 'move', 'after']].sample(frac=1)
#     test = df[r == 'test'][['before', 'move', 'after']].sample(frac=1)
    
#     try:
#         os.mkdir(f"{path}/{k}")
#     except OSError:
#         pass
    
#     train.to_csv(f"{path}/{k}/train_{train_size}.csv", index=False, header=False)
#     val.to_csv(f"{path}/{k}/val_{val_size}.csv", index=False, header=False)
#     test.to_csv(f"{path}/{k}/test_{test_size}.csv", index=False, header=False)
#     return train, val, test

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    path = "data/v2"
    try:
        os.mkdir(path)
    except OSError as error:
        pass
    train_sizes = np.array([30, 300, 3000, 30000])
    val_size = 15000
    test_size = 15000
    # train, test, val = get_datasets(path=path,  
    #     train_sizes=train_sizes, val_size=val_size, test_size=test_size, 
    #     method_func=SerializeMoveAsString_cyclic1,  process="reasoning")
    train, test, val = get_datasets(path=path,  
        train_sizes=train_sizes, val_size=val_size, test_size=test_size, 
        method_func=SerializeMoveAsString_v2)
        