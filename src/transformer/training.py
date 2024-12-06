import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformer_model import TransformerModel, generate_square_subsequent_mask
from dataset import QuixoDataset
from tokenization import token_to_id, id_to_token, PAD_ID, BOS_ID, vocab_size, decode_sequence
from tqdm import tqdm, trange
from timeit import default_timer as timer


def train_epoch(model, device, optimizer, criterion, dataloader, drawing=None):
    model.train()
    total_loss = 0
   
    for i, batch in enumerate(tqdm(dataloader)):
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        max_len = max(input_ids.shape[1], target_ids.shape[1])
        input_ids = nn.functional.pad(input_ids, (0, max_len - input_ids.shape[1]), value=PAD_ID)
        target_ids = nn.functional.pad(target_ids, (0, max_len - target_ids.shape[1]), value=PAD_ID)
        
        tgt_mask = generate_square_subsequent_mask(target_ids.size(1)-1, device)

        optimizer.zero_grad()
        output = model(input_ids, target_ids[:, :-1], tgt_mask=tgt_mask)
        output_dim = output.shape[-1]
        loss = criterion(output.view(-1, output_dim), target_ids[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if drawing is not None:
            drawing(curr_loss=loss.item())
        

        # if i % 20 == 0:
        #     print(f"\nInput ({i} batch, 0 element) :\n" + ''.join(decode_sequence(input_ids[0].numpy(), id_to_token)))
        #     print(f"\nOutput ({i} batch, 0 element) :\n" + ''.join(decode_sequence(list(output[0].argmax(-1).detach().numpy()), id_to_token)))
        #     print(f"\nTarget ({i} batch, 0 element) :\n" + ''.join(decode_sequence(target_ids[0, 1:].numpy(), id_to_token)))
    
    return total_loss / len(dataloader)


def run_train(model, device, optimizer, criterion, train_loader, num_epochs, drawing=None):
    model.to(device)
    for epoch in range(1, num_epochs+1):
        print((f"{7 * '-'} Epoch: {epoch} {7 * '-'}"))
        start_time = timer()
        train_loss = train_epoch(model, device, optimizer, criterion, train_loader, drawing=drawing)
        end_time = timer()
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s"))
        
    return model

def greedy_decode(model, device, src, max_len, start_ids=[BOS_ID]):
    src = src.to(device)
    memory = model.encode(src, src_mask=None)
    generated = torch.tensor(start_ids).repeat(src.shape[0], 1)
   
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = generate_square_subsequent_mask(generated.size(1), device)
        output = model.decode(generated, memory, tgt_mask)
        prob = model.fc_out(output[:, -1])
        predicted_ids = torch.argmax(prob, dim=1)
        generated = torch.cat([generated, predicted_ids.unsqueeze(-1)], dim=1)

    return generated

def generate_sequence(model, device, input_ids):
    model.eval()
    generated_ids = greedy_decode(model, device, src=input_ids, max_len=input_ids.shape[1])
    return generated_ids    

def calculate_accuracy(model, dataloader, device):
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            input_ids, target_ids = batch['input_ids'], batch['target_ids']
            generated_ids = generate_sequence(model, device, batch['input_ids'])
            if i % 5 == 0: 
                print(12 * '-' + f"Batch {i} on inference" + 12 * '-')
                print(f"\nInput :\n" + ''.join(decode_sequence(input_ids[0].numpy(), id_to_token)))
                print(f"\nGenerated :\n" + ''.join(decode_sequence(generated_ids[0].numpy(), id_to_token)))
                print(f"\nTarget :\n" + ''.join(decode_sequence(target_ids[0].numpy(), id_to_token)))
            for pred_seq, tgt_seq in zip(generated_ids, target_ids):
                tgt_mask = tgt_seq != PAD_ID
                pred_seq = pred_seq[:len(tgt_mask)][tgt_mask]
                tgt_seq = tgt_seq[tgt_mask]
                if torch.equal(pred_seq, tgt_seq):
                    correct_predictions += 1
                total_predictions += 1
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    return accuracy