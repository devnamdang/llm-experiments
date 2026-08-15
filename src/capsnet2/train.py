import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from datetime import datetime

def show_image(images, titles=[]):
    """Show images. Dimensions (C,H,W) or (batch_size,C,H,W)"""
    if len(images.shape)==3:
        images = images.unsqueeze(0) # (C,H,W) -> (1,C,H,W)
    n_images = images.shape[0]
    n_columns = min(n_images, 5)
    n_rows = (n_images-1) // 5 + 1
    _, axs = plt.subplots(n_rows, n_columns, 
                          sharex=True, sharey=True, squeeze=False, 
                          layout='constrained', figsize=(1.5*n_columns, 1.5*n_rows),
                          dpi=200,
                        )
    for i, ax in enumerate(axs.flat[:n_images]):
        img = images[i].movedim(0,-1)
        ax.imshow(img)
        ax.set_xticks(range(0, img.shape[1], 10))
        ax.set_yticks(range(0, img.shape[0], 10))
        if len(titles) > 0:
            ax.set_title(titles[i])
    
    for ax in axs.flat[n_images:]:
        ax.remove()
    plt.show()

def get_image(data, idx):
    img, _ = data[idx]
    return img

def calc_batch_reconstruction_loss(input_batch, output_batch):
    loss_fn = nn.MSELoss()
    return loss_fn(input_batch.flatten(), output_batch.flatten())

def get_digits(v):
    predicted_labels = torch.argmax(v, dim=-1) 
    return predicted_labels

def calc_accuracy(v, labels):
    return torch.sum(get_digits(v) == labels) / len(labels)

def calc_margin_loss(v_norm, label, lam=0.5, m_plus=0.9, m_min=0.1, eps=1e-8):
    """
    Args:
        v torch.Tensor: Tensor of shape (n_digits)
    Returns:
        torch.Tensor of dim 1
    """
    relu = nn.ReLU()
    margin_loss = 0
    for i in range(10):
        if i == label: # Actual label
            margin_loss += relu(m_plus-v_norm[i])**2 # Actual label
        else:
            margin_loss += lam * relu(v_norm[i]-m_min)**2
    return margin_loss

def calc_batch_margin_loss(v_norm, labels):
    """
    Args:
        v (torch.Tensor): Tensor of shape (batch_size, n_digits)
        labels (torch.Tensor): Tensor of shape (n_digits)
    Returns:
        torch.Tensor of dim 1
    """
    batch_size = v_norm.shape[0]
    total_margin_loss = 0
    for i, v in enumerate(v_norm):
        total_margin_loss += calc_margin_loss(v, labels[i])
    return total_margin_loss/batch_size

def calc_batch_loss(out_digit_cap, out_img, in_img, labels):
    margin_loss = calc_batch_margin_loss(out_digit_cap, labels)
    reconstruction_loss = calc_batch_reconstruction_loss(in_img, out_img)
    loss = margin_loss + 0.0005 * reconstruction_loss #margin_loss
    return loss, margin_loss, reconstruction_loss

def eval_model(model, dataloader, n_batches):
    model.eval()
    batch_nr = 0
    total_reconstruction_loss = 0
    accuracy = 0
    total_margin_loss = 0
    total_loss = 0
    for in_img, labels in dataloader:
        with torch.no_grad():
            out_digit_cap, out_img = model(in_img)
            accuracy += calc_accuracy(out_digit_cap, labels)
            loss, margin_loss, reconstruction_loss = calc_batch_loss(out_digit_cap, out_img, in_img, labels)
            total_reconstruction_loss += reconstruction_loss
            total_margin_loss += margin_loss
            total_loss += loss
        batch_nr += 1
        if batch_nr == n_batches:
            break
    avg_reconstruction_loss = total_reconstruction_loss / n_batches
    accuracy = accuracy / n_batches
    avg_margin_loss = total_margin_loss / n_batches
    avg_loss = total_loss / n_batches
    model.train()
    return accuracy, avg_reconstruction_loss, avg_margin_loss, avg_loss

def show_example_image(model, data):
    model.eval()
    image, label = data
    image = image.unsqueeze(dim=0)
    with torch.no_grad():
        v_norm, output_image = model(image)
    model.train()
    examples = torch.cat((image, output_image.detach()), dim=0)
    pred_labels = get_digits(v_norm)
    titles = [f"Label: {label}", f"label: {pred_labels[0].item()}"]
    show_image(examples, titles=titles)


def train_model(model, optimizer, train_loader, val_loader, epochs, eval_interval=50, eval_batches=20):
    start_time = datetime.now()
    model.train()

    batch_count = 0
    train_results = []
    val_results = []

    for epoch in range(epochs):
        for i_batch, batch in enumerate(train_loader):
            optimizer.zero_grad()
            img_batch, labels = batch
            out_digit_cap, out_img = model(img_batch)
            loss, _, _ = calc_batch_loss(out_digit_cap, out_img, img_batch, labels)
            loss.backward()
            optimizer.step()
            batch_count += 1

            # Track performance during training
            if (batch_count == 1) | (batch_count % eval_interval == 0) | (i_batch==len(train_loader)-1): 
                # Eval per interval, first and last batch
                accuracy, reconstruction_loss, margin_loss, combined_loss = eval_model(model, train_loader, eval_batches)
                print(f"Train set. epoch: {epoch:03d}, batch count: {batch_count:06d}, accuracy: {accuracy:.4f}, reconstruction loss: {reconstruction_loss:.4f}, margin loss: {margin_loss:.4f}, combined loss: {combined_loss:.4f}")
                train_results.append(
                    {
                        "epoch": epoch,
                        "batch_count": batch_count,
                        "accuracy": accuracy.item(),
                        "reconstruction_loss": reconstruction_loss.item(),
                        "margin_loss": margin_loss.item(),
                        "combined_loss": combined_loss.item()
                    }
                )

                accuracy, reconstruction_loss, margin_loss, combined_loss = eval_model(model, val_loader, eval_batches)
                print(f"Val set. epoch: {epoch:03d}, batch count: {batch_count:06d}, accuracy: {accuracy:.4f}, reconstruction loss: {reconstruction_loss:.4f}, margin loss: {margin_loss:.4f}, combined loss: {combined_loss:.4f}")
                val_results.append(
                    {
                        "epoch": epoch,
                        "batch_count": batch_count,
                        "accuracy": accuracy.item(),
                        "reconstruction_loss": reconstruction_loss.item(),
                        "margin_loss": margin_loss.item(),
                        "combined_loss": combined_loss.item()
                    }
                )

                show_example_image(model, val_loader.dataset[0])
                show_example_image(model, val_loader.dataset[1])

    end_time = datetime.now()
    print('Training duration: {}'.format(end_time - start_time))
    return train_results, val_results
