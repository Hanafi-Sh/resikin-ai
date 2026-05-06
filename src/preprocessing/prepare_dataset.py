"""
ResikIn Waste Classifier - Dataset Preparation
Download dataset dari Roboflow dan organize ke folder train/val/test.
"""

import os
import shutil
import argparse
from pathlib import Path


def organize_roboflow_dataset(roboflow_dir: str, output_dir: str):
    """
    Roboflow download biasanya sudah split train/valid/test.
    Script ini memastikan struktur folder sesuai kebutuhan training.
    
    Expected Roboflow structure:
        roboflow_dir/
        ├── train/
        │   ├── waste/
        │   └── not_waste/
        ├── valid/
        │   ├── waste/
        │   └── not_waste/
        └── test/
            ├── waste/
            └── not_waste/
    """
    splits = {"train": "train", "valid": "val", "test": "test"}
    
    for src_split, dst_split in splits.items():
        src_path = os.path.join(roboflow_dir, src_split)
        if not os.path.exists(src_path):
            print(f"⚠️  Split '{src_split}' not found, skipping...")
            continue
            
        for class_name in os.listdir(src_path):
            src_class = os.path.join(src_path, class_name)
            if not os.path.isdir(src_class):
                continue
                
            dst_class = os.path.join(output_dir, dst_split, class_name)
            os.makedirs(dst_class, exist_ok=True)
            
            count = 0
            for fname in os.listdir(src_class):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                    shutil.copy2(
                        os.path.join(src_class, fname),
                        os.path.join(dst_class, fname)
                    )
                    count += 1
            
            print(f"  [{dst_split}] {class_name}: {count} images")
    
    print("\n✅ Dataset siap digunakan untuk training!")


def print_stats(data_dir: str):
    """Print statistik dataset."""
    print("\n📊 Dataset Statistics:")
    print("-" * 40)
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                n = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))])
                print(f"  {split:6s} / {class_name:15s} : {n:5d} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roboflow_dir", type=str, required=True,
                        help="Path ke folder hasil download Roboflow")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory (default: ./data)")
    args = parser.parse_args()
    
    print("🔄 Organizing Roboflow dataset...")
    organize_roboflow_dataset(args.roboflow_dir, args.output_dir)
    print_stats(args.output_dir)
