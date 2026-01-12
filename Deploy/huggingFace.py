from huggingface_hub import HfApi
from dotenv import load_dotenv
import os


load_dotenv
api=HfApi(token=os.getenv("HF_TOKEN"))

for category in ["Accessories", "Laptop", "Smartphone", "Tablet"]:
    model_path = f"../Model/rf_model_{category}.joblib"
    repo_path = f"Model/rf_model_{category}.joblib"

    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=repo_path,
            repo_id="Sidikat123/Electrotech-Sales-Forecasting-Model",
            repo_type="model",
        )
        print(f" Uploaded {category}")
    except Exception as e:
        print(f" Failed to upload {category}: {str(e)}")
