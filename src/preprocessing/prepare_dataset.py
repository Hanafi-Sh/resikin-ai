"""
ResikIn Waste Classifier - Dataset Preparation
Download dataset dari Roboflow dan organize ke folder train/val/test.
"""

import os
import shutil
import argparse
from pathlib import Path


import random

def organize_roboflow_dataset(roboflow_dir: str, output_dir: str):
    """
    Roboflow download biasanya sudah split train/valid/test.
    Script ini memastikan struktur folder sesuai kebutuhan training.
    Jika valid/test tidak ada, script akan otomatis membagi data train (80/10/10).
    """
    splits = {"train": "train_data", "valid": "val_data", "test": "test_data"}
    has_valid_or_test = os.path.exists(os.path.join(roboflow_dir, "valid")) or os.path.exists(os.path.join(roboflow_dir, "test"))
    
    if not has_valid_or_test:
        print("⚠️  Split 'valid' atau 'test' tidak ditemukan di Roboflow. Akan melakukan auto-split 80/10/10 dari data train.")

    for src_split, dst_split in splits.items():
        src_path = os.path.join(roboflow_dir, src_split)
        
        if not os.path.exists(src_path):
            if has_valid_or_test:
                print(f"⚠️  Split '{src_split}' not found, skipping...")
            continue
            
        for class_name in os.listdir(src_path):
            src_class = os.path.join(src_path, class_name)
            if not os.path.isdir(src_class):
                continue
            
            # Kumpulkan semua gambar
            images = [f for f in os.listdir(src_class) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
            
            if not has_valid_or_test and src_split == "train":
                # Lakukan split manual
                random.shuffle(images)
                n_total = len(images)
                n_val = int(n_total * 0.1)
                n_test = int(n_total * 0.1)
                
                splits_data = {
                    "train_data": images[n_val + n_test:],
                    "val_data": images[:n_val],
                    "test_data": images[n_val:n_val + n_test]
                }
                
                for target_split, split_imgs in splits_data.items():
                    dst_class = os.path.join(output_dir, target_split, class_name)
                    os.makedirs(dst_class, exist_ok=True)
                    count = 0
                    for fname in split_imgs:
                        shutil.copy2(os.path.join(src_class, fname), os.path.join(dst_class, fname))
                        count += 1
                    print(f"  [{target_split}] {class_name}: {count} images")
            else:
                # Normal copy (kalau dataset dari sananya sudah ada split valid/test)
                dst_class = os.path.join(output_dir, dst_split, class_name)
                os.makedirs(dst_class, exist_ok=True)
                
                count = 0
                for fname in images:
                    shutil.copy2(os.path.join(src_class, fname), os.path.join(dst_class, fname))
                    count += 1
                print(f"  [{dst_split}] {class_name}: {count} images")
    
    print("\n✅ Dataset siap digunakan untuk training!")


def print_stats(data_dir: str):
    """Print statistik dataset."""
    print("\n📊 Dataset Statistics:")
    print("-" * 40)
    for split in ["train_data", "val_data", "test_data"]:
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
