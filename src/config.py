import yaml
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"

def load_config(path: str = None):
    p = Path(path) if path else DEFAULT_PATH
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
