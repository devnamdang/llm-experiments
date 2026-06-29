#%%
import json
import os
import time
from functools import partial
from pathlib import Path

import yaml
import tiktoken
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.gpt2.model import calc_loss_loader, train_model_simple
from src.gpt2.train import plot_losses
from src.gpt2.load import load_model
import src.utils as ut

#%%

CONFIG_PATH = Path('config/config_instruction.yaml')

def get_configs():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    training_config = config['training']
    model_config = config['model']
    optimizer_config = config['optimizer']
    data_config = config['data']
    return training_config, model_config, optimizer_config, data_config

def load_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    #print("Number of entries:", len(data))
    return data

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

def load_data(instruction_data_path, train_test_portion):
    assert train_test_portion[0] + train_test_portion[1] < 1, "Sum of train test portion must be smaller than 1"
    data = load_json_file(instruction_data_path)
    # Split data
    train_portion = int(len(data) * train_test_portion[0])
    test_portion = int(len(data) * train_test_portion[1])
    train_data = data[:train_portion]
    test_data = data[train_portion:(train_portion+test_portion)]
    val_data = data[(train_portion+test_portion):]
    return train_data, test_data, val_data


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        
        for entry in data:
            instruction_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))
    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def custom_collate_fn(batch, pad_token_id=50256, device='cpu', 
                           allowed_max_length=None, ignore_index=-100):
    batch_max_length = max(len(item)+1 for item in batch)
    input_lst, target_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        nr_pads = batch_max_length - len(new_item)
        padded = new_item + [pad_token_id] * nr_pads

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = (targets == pad_token_id)
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        input_lst.append(inputs)
        target_lst.append(targets)

    input_tensor = torch.stack(input_lst).to(device)
    target_tensor = torch.stack(target_lst).to(device)
    return input_tensor, target_tensor

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_data_loaders(data_config, training_config, tokenizer, customized_collate_fn):
    train_data, test_data, val_data = load_data(data_config['path'], training_config["train_test_fraction"])
    num_workers = training_config['num_workers']
    batch_size = training_config['batch_size']

    torch.manual_seed(training_config['manual_seed'])
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    return train_loader, val_loader, test_loader, train_data, test_data, val_data

def run():
    training_config, model_config, optimizer_config, data_config = get_configs()
    model, optimizer, gpt_config = load_model(model_config['foundation_model'])

    device = get_device()
    customized_collate_fn = partial(
        custom_collate_fn, 
        device=device,
        allowed_max_length=gpt_config['model']['context_length']
    )

    tokenizer = tiktoken.get_encoding(gpt_config['model']['tokenizer'])
    tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
    train_loader, val_loader, test_loader, _, _, val_data = load_data_loaders(data_config, training_config, tokenizer, customized_collate_fn)

    # %%
    start_time = time.time()
    torch.manual_seed(training_config['manual_seed'])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=optimizer_config['lr'], weight_decay=optimizer_config['weight_decay']
    )

    num_epochs = training_config['epochs']
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=training_config['eval']['freq'], eval_iter=training_config['eval']['iter'],
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    #%% Save output
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    losses_fig = plot_losses(epochs_tensor.tolist(), tokens_seen, train_losses, val_losses)
    losses_df = pd.DataFrame({
        "epochs": epochs_tensor.tolist(),
        "tokens_seen": tokens_seen,
        "train_losses": train_losses,
        "val_losses": val_losses
    })

    calc_loss_loader_partial = partial(calc_loss_loader, model=model, device=device, num_batches=10)
    final_loss = [calc_loss_loader_partial(loader) for loader in [train_loader, val_loader, test_loader]]
    final_loss_df = pd.DataFrame({
        'data_set': ['train', 'validation', 'test'],
        'loss': final_loss
    })

    performance_dict = {
        'losses': losses_df,
        'final_loss': final_loss_df
    }
    fig_dict = {'losses': losses_fig}

    ut.save_output(model, optimizer, fig_dict, performance_dict, CONFIG_PATH, "instruction")

# %%
if __name__ == "__main__":
    run()