import os
import shutil
import datetime
import torch
from pathlib import Path

OUTPUT_DIR = Path("output")


def save_output(model, optimizer, fig_dict, performance_dict, config_path, prefix):
    """
    Saves output of a training session
    """
    # Create output dir
    output_run_dir = OUTPUT_DIR / (prefix + "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    os.mkdir(output_run_dir)

    # Copy the config file
    shutil.copyfile(config_path, output_run_dir / config_path.name)

    # Save model and optimizer
    save_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    mdl_path = output_run_dir / "model_and_optimizer.pth"
    torch.save(save_dict, mdl_path)

    # Save performance results
    save_df = lambda df, name: df.to_csv(output_run_dir / name)
    for key in performance_dict.keys():
        save_df(performance_dict[key], key)

    # Save figures
    save_fig = lambda fig, name: fig.savefig(output_run_dir / name)
    for key in fig_dict.keys():
        save_fig(fig_dict[key], key)