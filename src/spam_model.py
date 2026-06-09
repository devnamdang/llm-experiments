import torch

def transform_to_classification_model(model, num_classes=2):
    """
    
    args:
        model: GPT foundation model
    
    returns
        classificatiom model for training
    """

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Change out head
    model.out_head = torch.nn.Linear(
        in_features = model.out_head.in_features,
        out_features = num_classes
    )

    # Make last transformer block and layer norm trainable
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    return model

def classify_review(
        text, model, tokenizer, device, max_length=None, pad_token_id = 50256
    ):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]

    input_ids = input_ids[:min(max_length, supported_context_length)]
    input_ids += [pad_token_id] * (max_length-len(input_ids))

    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:,-1,:]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"