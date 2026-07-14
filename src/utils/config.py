import yaml


def load_train_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_loss_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
