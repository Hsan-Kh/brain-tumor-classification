"""
src/data_setup.py
Responsible for creating DataLoaders and applying transforms.
Adheres to: Single Responsibility Principle.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int = 32
):
    """
    Creates training and testing DataLoaders.
    
    Args:
        train_dir: Path to training data.
        test_dir: Path to testing data.
        transform: Composition of transforms to apply to images.
        batch_size: Number of samples per batch.

    Returns:
        train_dataloader, test_dataloader, class_names
    """
    
    # 1. Validation: Ensure paths exist
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"[CRITICAL] Data directories not found. Check: {train_dir}")

    # 2. Use ImageFolder (Standard for structure: folder_name = class_name)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_data.classes
    print(f"[INFO] Classes Found: {class_names}")

    # 3. Create DataLoaders
    # pin_memory=True speeds up transfer from CPU to RAM
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True, # Shuffle training data to prevent order bias
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False, # Don't shuffle test data
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names