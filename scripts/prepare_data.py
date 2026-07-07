import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    output_dir = "output"
    work_code = "HVB_001"
    work_dir = os.path.join(output_dir, work_code)
    
    if not os.path.exists(work_dir):
        print(f"[Error] Aligned work directory not found: {work_dir}. Please run run_mapping.py first.")
        return
        
    print(f"Gathering aligned parallel TSV files from {work_dir}...")
    
    all_pairs = []
    
    # Traverse directory to find all _parallel.tsv files
    for root, dirs, files in os.walk(work_dir):
        for file in files:
            if file.endswith("_parallel.tsv"):
                tsv_path = os.path.join(root, file)
                print(f"  Reading: {tsv_path}")
                try:
                    df = pd.read_csv(tsv_path, sep="\t")
                    # Ensure columns exist and drop empty ones
                    df = df.dropna(subset=["han_sentence", "viet_sentence"])
                    for _, row in df.iterrows():
                        han = str(row["han_sentence"]).strip()
                        viet = str(row["viet_sentence"]).strip()
                        if han and viet:
                            all_pairs.append({
                                "translation": {
                                    "zh": han,
                                    "vi": viet
                                }
                            })
                except Exception as e:
                    print(f"  Error reading {tsv_path}: {e}")
                    
    total_pairs = len(all_pairs)
    print(f"Total valid parallel sentence pairs gathered: {total_pairs}")
    
    if total_pairs == 0:
        print("[Error] No parallel sentences found. Cannot prepare dataset.")
        return
        
    # Split into train and validation sets (90% train, 10% validation)
    train_data, val_data = train_test_split(all_pairs, test_size=0.1, random_state=42)
    
    dataset_dir = os.path.join(output_dir, "translation_dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    train_path = os.path.join(dataset_dir, "train.json")
    val_path = os.path.join(dataset_dir, "val.json")
    
    # Save train set
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # Save val set
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Dataset preparation complete:")
    print(f"  Train set: {len(train_data)} samples saved to {train_path}")
    print(f"  Validation set: {len(val_data)} samples saved to {val_path}")

if __name__ == "__main__":
    main()
