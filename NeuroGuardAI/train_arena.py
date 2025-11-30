"""
train_arena.py
Entry point. Trains multiple models sequentially and saves them.
"""
import torch
import torch.nn as nn
from torchvision import transforms
from src import data_setup, model_builder, engine
import os
import time

# --- CONFIGURATION ---
NUM_EPOCHS = 5 
BATCH_SIZE = 32
LEARNING_RATE = 0.001


MODELS_TO_TRAIN = ["resnet18", "mobilenet", "efficientnet", "densenet121"]

DATA_PATH = "data"
TRAIN_DIR = f"{DATA_PATH}/Training"
TEST_DIR = f"{DATA_PATH}/Testing"
SAVE_DIR = "models"

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SYSTEM] Running on: {device}")

    train_dl, test_dl, class_names = data_setup.create_dataloaders(
        TRAIN_DIR, TEST_DIR, data_transform, BATCH_SIZE
    )
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    for model_name in MODELS_TO_TRAIN:
        print(f"\n{'='*20} TRAINING {model_name.upper()} {'='*20}")
        
        model = model_builder.build_model(model_name, len(class_names), device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        start_time = time.time()

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = engine.train_step(model, train_dl, loss_fn, optimizer, device)
            test_loss, test_acc = engine.test_step(model, test_dl, loss_fn, device)
            
            print(f"Epoch: {epoch+1} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        end_time = time.time()
        print(f"[DONE] {model_name} finished in {end_time - start_time:.2f}s")
        
        save_path = os.path.join(SAVE_DIR, f"{model_name}_brain_tumor.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[SAVED] Model saved to {save_path}")

if __name__ == "__main__":
    main()