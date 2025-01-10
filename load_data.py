import pickle
import pandas as pd
from typing import Dict, Any

def load_pickle(pickle_path: str) -> Dict[str, Any] | None:
    try:
        with open(pickle_path, "rb") as file:
            data = pickle.load(file)
            print(f"[INFO] Data successfully loaded. Data type: {type(data)}")
        return data
    except Exception as e:
        print(f"[ERROR] Exception occurred while loading pickle: {e}")
        return None

def flatten_data(data: Dict[str, Any]) -> pd.DataFrame | None:
    if not isinstance(data, dict):
        print("[ERROR] Invalid data type. Expected a dictionary.")
        return None

    flattened_data = [
        {
            "syndrome_id": syndrome_id,
            "subject_id": subject_id,
            "image_id": image_id,
            "embedding": embedding,
        }
        for syndrome_id, subjects in data.items()
        for subject_id, images in subjects.items()
        for image_id, embedding in images.items()
    ]
    df = pd.DataFrame(flattened_data)
    print("[INFO] Data successfully flattened into a DataFrame.")
    return df
