import os
# Ensure the HF_HOME environment variable points to your desired cache location
os.environ["HF_TOKEN"] = "Your HF Token"
cache_dir = 'Your Cache Directory'
os.environ['HF_HOME'] = cache_dir
import json
from datasets import load_dataset


# Set dataset save path
save_path = "cnn.json"
if not os.path.exists(save_path):
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    train_data = dataset["train"][:50000]
    test_data = dataset["test"][:1000]
    
    data_subset = []
    
    for article, highlights, data_id in zip(train_data["article"], train_data["highlights"], train_data["id"]): 
        data_subset.append({
            "id": data_id,
            "article": article,
            "highlights": highlights,
            "type": "train"
        })
    
    for article, highlights, data_id in zip(test_data["article"], test_data["highlights"], test_data["id"]):
        data_subset.append({
            "id": data_id,
            "article": article,
            "highlights": highlights,
            "type": "test"
        })
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data_subset, f, ensure_ascii=False, indent=4)

print(f"Data saved to {save_path}")