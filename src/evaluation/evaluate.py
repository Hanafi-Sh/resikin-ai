"""
ResikIn Waste Classifier - Evaluation Script
Evaluasi model fine-tuned pada test set dan generate classification report.
"""

import os
import json
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import argparse


def load_test_data(test_dir):
    samples = []
    class_names = sorted(os.listdir(test_dir))
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in os.listdir(cls_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                samples.append((os.path.join(cls_dir, f), idx))
    return samples, class_names


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"✅ Loaded fine-tuned weights: {args.model_path}")
    model.to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    text_labels = [
        "a photo of garbage, trash, or waste",
        "a photo that is not garbage or waste",
    ]

    # Load test data
    samples, class_names = load_test_data(args.test_dir)
    print(f"Test samples: {len(samples)}, Classes: {class_names}")

    all_preds = []
    all_labels = []

    for img_path, label in tqdm(samples, desc="Testing"):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.logits_per_image.argmax(dim=1).item()

        all_preds.append(pred)
        all_labels.append(label)

    # Results
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n{'='*50}")
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*50}")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Save results
    results = {
        "accuracy": acc,
        "report": classification_report(all_labels, all_preds, target_names=class_names, output_dict=True),
        "confusion_matrix": cm.tolist(),
    }
    out_path = os.path.join(os.path.dirname(args.model_path), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="./data/test")
    parser.add_argument("--model_path", type=str, default="./models/clip_waste_classifier.pth")
    args = parser.parse_args()
    main(args)
