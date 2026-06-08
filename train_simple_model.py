#%%
import tiktoken
import matplotlib.pyplot as plt
import torch
import yaml
import os
from os.path import isfile, join

import shutil
import datetime
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from matplotlib.ticker import MaxNLocator
import src.gpt as gpt
import src.data as dt

#%% Constants
DATA_DIR = Path("/Volumes/ext-2-1000GB/code/python/gutenberg/data/raw")
CONFIG_PATH = Path("config.yaml")
OUTPUT_DIR = Path("output")
MAX_FILES = 1

#%% Config
with open(CONFIG_PATH, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
training_config = config['training']
model_config = config['model']
optimizer_config = config['optimizer']

#%% Load data

def load_text_files():
    files = [f for f in os.listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]
    text_list = []
    for cnt, file_path in enumerate(files):
        with open(DATA_DIR/file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        text_list.append(raw_text)
        text_list.append('<|endoftext|>')
        if cnt == MAX_FILES:
            break
    text = ' '.join(text_list)
    return text

def create_data_loaders():

    raw_text = load_text_files()

    # Train/test split
    train_ratio = training_config['train_ratio']
    split_idx = int(train_ratio * len(raw_text))
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]

    train_loader = dt.create_data_loader_v1(
        train_data,
        batch_size=training_config['batch_size'],
        max_length=model_config["context_length"],
        stride=model_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    if train_ratio < 1.0:
        val_loader = dt.create_data_loader_v1(
            val_data,
            batch_size=training_config['batch_size'],
            max_length=model_config["context_length"],
            stride=model_config["context_length"],
            drop_last=False,
            shuffle=False,
            num_workers=0
        )
    else:
        val_loader = None

    return train_loader, val_loader


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3), dpi=300)
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
    epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    return fig

def save_output(model, optimizer, fig, losses_df):
    # Create output dir
    output_run_dir = OUTPUT_DIR / datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    os.mkdir(output_run_dir)

    # Copy the config file
    shutil.copyfile(CONFIG_PATH, output_run_dir / CONFIG_PATH.name)

    # Save model and optimizer
    save_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    mdl_path = output_run_dir / "model_and_optimizer.pth"
    torch.save(save_dict, mdl_path)

    # Save loss resuls
    losses_df.to_csv(output_run_dir / "losses.csv")

    # Save figures
    fig.savefig(output_run_dir / "losses.png")

#%%
def main():
    train_loader, val_loader = create_data_loaders()
    torch.manual_seed(training_config['manual_seed'])
    model =gpt.GPTModel(model_config)
    tokenizer = tiktoken.get_encoding(model_config['tokenizer'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
            lr=optimizer_config['lr'], weight_decay=optimizer_config['weight_decay']
    )

    # Train model
    num_epochs = training_config['num_epochs']
    train_losses, val_losses, tokens_seen = gpt.train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs,
        eval_freq=training_config['eval']['freq'],
        eval_iter=training_config['eval']['iter'],
        start_context=training_config['eval']['start_context'],
        tokenizer=tokenizer
    )

    # Create losses plot
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    losses_fig = plot_losses(epochs_tensor.tolist(), tokens_seen, train_losses, val_losses)
    losses_df = pd.DataFrame({
        "epochs": epochs_tensor.tolist(),
        "tokens_seen": tokens_seen,
        "train_losses": train_losses,
        "val_losses": val_losses
    })

    # Finish up
    save_output(model, optimizer, losses_fig, losses_df)


if __name__ == "__main__":
    main()

