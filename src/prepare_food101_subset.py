import torch
from data import get_dataloaders
from model import create_model
from engine import train_one_epoch, validate
from utils import save_model


def main():
    # -------------------------
    # Config
    # -------------------------
    DATA_DIR = "data"
    BATCH_SIZE = 32
    IMG_SIZE = 224
    EPOCHS = 8
    LR = 3e-4

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", DEVICE)

    # -------------------------
    # Data
    # -------------------------
    train_loader, val_loader, classes = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE
    )

    print("Num classes:", len(classes))

    # -------------------------
    # Model
    # -------------------------
    model = create_model(num_classes=len(classes)).to(DEVICE)

    # -------------------------
    # Optim / Loss / AMP
    # -------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=DEVICE,
            epoch=epoch
        )

        validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=DEVICE
        )

        save_model(model, f"models/epoch_{epoch+1}.pth")


# -------------------------
# REQUIRED FOR WINDOWS
# -------------------------
if __name__ == "__main__":
    main()
