import yaml

def load_config(path: str)-> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def build_run_name(config: dict) -> str:
    dataset_name = config["dataset"].get("name", "lasa").lower()
    pattern = config["dataset"]["pattern_name"]
    hidden = config["model"]["hidden_dim"]
    lr = config["training"]["lr"]
    name = f"{dataset_name}_{pattern}_h{hidden}_lr{lr}"
    if config['dataset'].get('use_velocity', False):
        name += "_vel"
    if config['dataset'].get('use_acceleration', False):
        name += "_acc"
    return name
