tokens = ['<PAD>', '<BOS>', '<EOS>', '<SEP>', 'X', 'O', '#', '\n', '0', '1', '2', '3', '4', 'up', 'down', 'left', 'right']
token_to_id = {token: idx for idx, token in enumerate(tokens)}
id_to_token = {idx: token for idx, token in enumerate(tokens)}
vocab_size = len(tokens)
PAD_ID, BOS_ID, EOS_ID, X_ID, O_ID = token_to_id['<PAD>'], token_to_id['<BOS>'], token_to_id['<EOS>'], token_to_id['X'], token_to_id['O']


def create_input_sequence(initial_state_str, move_str):
    initial_seq = list(initial_state_str)
    move_seq = move_str.strip().split()
    return initial_seq + ['<SEP>'] + move_seq + ['<EOS>']

def tokenize_target_state(target_state_str):
    sequence = ['<BOS>'] + list(target_state_str) + ['<EOS>']
    return sequence

def encode_sequence(sequence, token_to_id):
    return [token_to_id[token] for token in sequence]

def decode_sequence(sequence, id_to_token):
    return [id_to_token[id] for id in sequence]
