import torch
from generator import BOARD_SIZE 

def state_string_to_board(state_str):
    board = state_str.strip().split('\n')[:5]
    return board


def preprocess_state_and_move(state_str, move_str):
    board = state_string_to_board(state_str)
    char_to_idx = {'X': 0, 'O': 1, '#': 2}
    row, col, direction, piece = move_str.strip().split()
    row = int(row)
    col = int(col)
    end_point = 0 if direction in ["down", "right"] else BOARD_SIZE - 1
    new_row, new_col = row, col
    if direction in ["left", "right"]:
        new_col = end_point
    else:
        new_row = end_point

    state_tensor = torch.zeros(2, 5, 5)
    for i in range(5):
        for j in range(5):
            char = board[i][j]
            idx = char_to_idx[char]
            if idx == 2:
                continue
            state_tensor[idx, i, j] = 1

    move_tensor = torch.zeros(3, 5, 5)
    move_tensor[2][row][col] = 1
    idx = char_to_idx[piece]
    move_tensor[idx][new_row][new_col] = 1
    return state_tensor, move_tensor
    
def preprocess_state(state_str):
    board = state_string_to_board(state_str)
    char_to_idx = {'X': 0, 'O': 1, '#': 2}
    state_tensor = torch.zeros(2, 5, 5)
    for i in range(5):
        for j in range(5):
            char = board[i][j]
            idx = char_to_idx[char]
            if idx == 2:
                continue
            state_tensor[idx, i, j] = 1
    return state_tensor
    

def tensor_to_string(state_tensor):
    state_str = ""
    chars = ['X', 'O']
    for i in range(5):
        for j in range(5):
            char = "#"
            for k in range(2):
                if state_tensor[k][i][j] == 1:
                    char = chars[k]
            state_str += char
        state_str += "\n"
    return state_str
  