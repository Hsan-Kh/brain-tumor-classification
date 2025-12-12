"""
src/engine.py
Contains train_step and test_step functions.
"""
import torch
from tqdm.auto import tqdm

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train() # Set to training mode (enables Dropout/BatchNorm)
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(tqdm(dataloader, desc="Training")):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    return train_loss / len(dataloader), train_acc / len(dataloader)

def test_step(model, dataloader, loss_fn, device):
    model.eval() # Set to evaluation mode
    test_loss, test_acc = 0, 0

    with torch.inference_mode(): # Context manager (faster than no_grad)
        for X, y in tqdm(dataloader, desc="Testing"):
            X, y = X.to(device), y.to(device)
            
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            
            test_pred_labels = test_pred.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    return test_loss / len(dataloader), test_acc / len(dataloader)