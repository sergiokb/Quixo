import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataset import QuixoDataset
from training import run_train, calculate_accuracy, TransformerModel, QuixoDataset
from tokenization import vocab_size, PAD_ID


if __name__ == "__main__":
    model_path = "model_1.pth"
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 10
    
    
    size = 640
    path = "/../data/v2"
    train_data = pd.read_csv(f"{path}/3_train_30000.csv")[:size]
    val_data = pd.read_csv(f"{path}/val_15000.csv")[:size]
    test_data = pd.read_csv(f"{path}/test_15000.csv")[:size]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TransformerModel(
        d_model = 512,
        nhead = 8,
        num_encoder_layers = 6,
        num_decoder_layers = 6,
        dim_feedforward = 2048,
        dropout = 0.1,
        vocab_size=vocab_size
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion=nn.CrossEntropyLoss(ignore_index=PAD_ID)
    
   
    train_dataset = QuixoDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   
    model = run_train(model=model, 
                      device=device,  
                      optimizer=optimizer,
                      criterion=criterion,
                      train_loader=train_loader,
                      num_epochs=num_epochs)
    
    
    torch.save(model.state_dict(), model_path)
    
    test_dataset = QuixoDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    accuracy = calculate_accuracy(model, test_loader, device)
    print(f"Test Accuracy: {accuracy}")