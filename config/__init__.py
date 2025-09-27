import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent

# Load JSON once
with open(CONFIG_PATH / "params.json") as f:
    params = json.load(f)

# Also load materials
with open(CONFIG_PATH / "materials.json") as f:
    materials = json.load(f)
