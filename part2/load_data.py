import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt', quiet=True)
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.encoder_inputs, self.decoder_inputs = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)

        encoder_inputs = []
        for nl in nl_lines:
            enc = tokenizer(nl, truncation=True, max_length=128)
            encoder_inputs.append(torch.tensor(enc['input_ids'], dtype=torch.long))

        if split != 'test':
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_lines = load_lines(sql_path)
            decoder_inputs = []
            for sql in sql_lines:
                # Tokenizer appends EOS (id=1) automatically
                dec = tokenizer(sql, truncation=True, max_length=512)
                decoder_inputs.append(torch.tensor(dec['input_ids'], dtype=torch.long))
            return encoder_inputs, decoder_inputs
        else:
            return encoder_inputs, None

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        if self.split != 'test':
            return self.encoder_inputs[idx], self.decoder_inputs[idx]
        else:
            return (self.encoder_inputs[idx],)

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    decoder_input  = [PAD/BOS, t1, t2, ..., t_{n-1}]   (PAD as T5 decoder start token)
    decoder_target = [t1, t2, ..., t_{n-1}, EOS]        (shifted left by one)
    '''
    enc_ids = [item[0] for item in batch]
    dec_ids = [item[1] for item in batch]

    encoder_ids = pad_sequence(enc_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    dec_inputs_list = []
    dec_targets_list = []
    for dec in dec_ids:
        # dec = [t1, t2, ..., tn, EOS=1]
        # decoder input: prepend PAD as BOS, drop EOS from input side
        dec_input = torch.cat([torch.tensor([PAD_IDX], dtype=torch.long), dec[:-1]])
        dec_target = dec  # full sequence including EOS as supervision
        dec_inputs_list.append(dec_input)
        dec_targets_list.append(dec_target)

    decoder_inputs = pad_sequence(dec_inputs_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(dec_targets_list, batch_first=True, padding_value=PAD_IDX)

    # Initial decoder input for autoregressive generation: single PAD token per example
    initial_decoder_inputs = torch.zeros(len(batch), 1, dtype=torch.long)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.
    '''
    enc_ids = [item[0] for item in batch]

    encoder_ids = pad_sequence(enc_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    initial_decoder_inputs = torch.zeros(len(batch), 1, dtype=torch.long)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")

    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x
