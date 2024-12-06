import torch
from torch.utils.data import Dataset, DataLoader
from tokenization import encode_sequence, create_input_sequence, tokenize_target_state, token_to_id


class QuixoDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        initial_state = self.data.iloc[idx, 0]
        move = self.data.iloc[idx, 1]
        target_state = self.data.iloc[idx, 2]
        
        input_seq = create_input_sequence(initial_state, move)
        target_seq = tokenize_target_state(target_state)
        
        input_ids = encode_sequence(input_seq, token_to_id)
        target_ids = encode_sequence(target_seq, token_to_id)
        
        return {'input_ids': torch.tensor(input_ids), 'target_ids': torch.tensor(target_ids)}