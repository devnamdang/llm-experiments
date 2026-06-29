#%%
from src.gpt2 import model
import torch
import yaml
from pathlib import Path

OUTPUT_DIR = Path("output")
# %%

def load_model(model_name):
    # Import config
    model_path = OUTPUT_DIR / model_name
    model_architecture = model_name.split("_")[0]
    config_path = model_path / f"config_{model_architecture}.yaml"
    checkpoint_path = model_path / "model_and_optimizer.pth"

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    model_config = config['model']

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = model.GPTModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, config

