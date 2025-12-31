from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

def get_transforms(train=True, img_size=224):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

def get_dataloaders(data_dir, batch_size=32, img_size=224):
    data_dir = Path(data_dir)

    train_ds = datasets.ImageFolder(
        data_dir / "train",
        transform=get_transforms(train=True, img_size=img_size)
    )

    val_ds = datasets.ImageFolder(
        data_dir / "val",
        transform=get_transforms(train=False, img_size=img_size)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0   # IMPORTANT for Windows
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, train_ds.classes
