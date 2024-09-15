import json
from pathlib import Path

# Load configuration from config.json
CONFIG_PATH = Path(__file__).parent / "config.json"

with open(CONFIG_PATH, "r") as config_file:
    CONFIG = json.load(config_file)

HUGGINGFACE_TOKEN = CONFIG["huggingfacehub_api_token"]
REPO_ID = CONFIG["repo_id"]
QDRANT_PATH = CONFIG["qdrant_path"]
QDRANT_COLLECTION_NAME = CONFIG["qdrant_collection_name"]
