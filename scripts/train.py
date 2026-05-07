"""
ResikIn Waste Classifier - Training Script
Fine-tune CLIP ViT-B/32 secara contrastive untuk klasifikasi sampah.

Cara pakai di Google Colab:
1. Upload script ini atau clone repo
2. Jalankan: python scripts/train.py --data_dir ./data --epochs 10 --batch_size 32
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt


# ── Label teks untuk contrastive learning ──
# Setiap kategori dipasangkan dengan deskripsi teks yang akan dilatih bersama image encoder
CATEGORY_PROMPTS = {
    "waste": [
        "a photo of garbage or waste on the ground",
        "a pile of trash and litter in a public area",
        "an overflowing garbage bin full of waste",
        "illegal dumping of trash on the street",
        "uncollected garbage bags on the sidewalk",
        "dirty polluted environment with scattered waste",
    ],
    "not_waste": [
        "a clean street with no garbage",
        "a selfie photo of a person smiling",
        "a photo of a pet cat or dog",
        "food and drinks on a dining table",
        "a beautiful park or natural scenery",
        "an indoor room with clean furniture",
    ],
}

# Mapping ke kategori ResikIn
RESIKIN_CATEGORIES = {
    "waste": ["tps_penuh", "sampah_liar", "tidak_terangkut"],
    "not_waste": ["spam"],
}


class WasteDataset(Dataset):
    """Dataset loader untuk gambar sampah dari folder structure."""

    def __init__(self, root_dir, processor, split="train"):
        self.processor = processor
        self.samples = []  # (image_path, label_idx)
        self.class_names = sorted(os.listdir(root_dir))
        
        for idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.samples.append((os.path.join(class_dir, fname), idx))

        print(f"[{split}] Loaded {len(self.samples)} images, {len(self.class_names)} classes: {self.class_names}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return image, label


def collate_fn(batch, processor, text_labels):
    """Custom collate: process images + text labels together for CLIP."""
    images, labels = zip(*batch)
    inputs = processor(
        text=text_labels,
        images=list(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return inputs, torch.tensor(labels)


def train_one_epoch(model, dataloader, optimizer, device, processor, text_labels):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_images, batch_labels in tqdm(dataloader, desc="Training"):
        # Process images and text
        inputs = processor(
            text=text_labels,
            images=list(batch_images),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        labels = batch_labels.to(device)

        # Forward pass - get similarity scores
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # (batch, num_text_labels)

        # Contrastive loss: image should match its correct text label
        loss = nn.CrossEntropyLoss()(logits_per_image, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits_per_image.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, device, processor, text_labels):
    model.eval()
    all_preds = []
    all_labels = []

    for batch_images, batch_labels in tqdm(dataloader, desc="Evaluating"):
        inputs = processor(
            text=text_labels,
            images=list(batch_images),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        outputs = model(**inputs)
        preds = outputs.logits_per_image.argmax(dim=1).cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(batch_labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["waste", "not_waste"], output_dict=True)
    return acc, report


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load CLIP
    print("Loading CLIP ViT-B/32...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Freeze most layers, only fine-tune last 2 layers of vision + text encoder
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze last 2 vision transformer layers + text projection + visual projection
    for name, param in model.named_parameters():
        if any(k in name for k in [
            "vision_model.encoder.layers.10",
            "vision_model.encoder.layers.11",
            "text_model.encoder.layers.10",
            "text_model.encoder.layers.11",
            "visual_projection",
            "text_projection",
        ]):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    # Text labels for contrastive matching
    text_labels = [
        "a photo of garbage, trash, or waste",
        "a photo that is not garbage or waste",
    ]

    # Datasets
    train_ds = WasteDataset(os.path.join(args.data_dir, "train_data"), processor, "train")
    val_ds = WasteDataset(os.path.join(args.data_dir, "val_data"), processor, "val")
    test_ds = WasteDataset(os.path.join(args.data_dir, "test_data"), processor, "test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: (list(zip(*b))[0], torch.tensor(list(zip(*b))[1])))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: (list(zip(*b))[0], torch.tensor(list(zip(*b))[1])))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda b: (list(zip(*b))[0], torch.tensor(list(zip(*b))[1])))

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01
    )

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")

        loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, processor, text_labels)
        val_acc, val_report = evaluate(model, val_loader, device, processor, text_labels)

        history["train_loss"].append(loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Final evaluation on test set
    print(f"\n{'='*50}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*50}")
    test_acc, test_report = evaluate(model, test_loader, device, processor, text_labels)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(classification_report(
        [s[1] for s in test_ds.samples],
        [outputs for outputs in range(len(test_ds))],  # placeholder
        target_names=["waste", "not_waste"]
    ) if False else f"Test report: {json.dumps(test_report, indent=2)}")

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "clip_waste_classifier.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=150)
    print(f"Training curves saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    main(args)
