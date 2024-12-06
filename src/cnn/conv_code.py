import pandas as pd
import numpy as np
import os 
from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from conv_preprocessing import preprocess_state, preprocess_state_and_move
from conv_preprocessing import tensor_to_string


class ConvDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        initial_state_str = self.data.iloc[idx, 0]
        move_str = self.data.iloc[idx, 1]
        target_state_str = self.data.iloc[idx, 2]
        
        initial_state, move = preprocess_state_and_move(initial_state_str, move_str)
        target_state = preprocess_state(target_state_str)
        
        return initial_state, move, target_state

class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(5, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 2, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, state, move):
        x = torch.cat((state, move), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x

def conv_run(sizes=[30, 300, 3000, 30000]):
    val_data = pd.read_csv("data/v2/val_15000.csv")
    test_data = pd.read_csv("data/v2/test_15000.csv")
    
    val_dataset = ConvDataset(val_data)
    test_dataset = ConvDataset(test_data)

    val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)
    result = {}

    for i, size in enumerate(sizes):
        train_data = pd.read_csv(f"data/v2/{i}_train_{size}.csv")
        train_dataset = ConvDataset(train_data)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
        
    
        model = ConvModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        num_epochs = 20
        
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                initial_states, moves, target_states = batch
                optimizer.zero_grad()
                outputs = model(initial_states, moves)
                loss = criterion(outputs, target_states.float())
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            correct = 0
            cnt = 0
            for batch in val_dataloader:
                initial_states, moves, target_states = batch
                outputs = model(initial_states, moves)
                b = ((outputs > 0.5).to(float) == target_states.float())
                for i in range(b.shape[0]):
                    if b[i].all():
                        correct += 1
                cnt += b.shape[0]
        
        assert cnt == len(val_dataset)
        accuracy = correct / len(val_dataset)
        result[f"train_{size}"] = accuracy
        
    return result

if __name__ == "__main__":
    # np.random.seed(42)
    # random.seed(42)
    result = conv_run(sizes=[30, 300, 3000, 30000]) # {'train_30': 0.1014, 'train_300': 0.2319, 'train_3000': 0.5805, 'train_30000': 0.7009}
    print(result)