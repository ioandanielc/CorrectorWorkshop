import logging
from datetime import datetime
from pathlib import Path


def create_run_dir(base_dir: str = "training_artifacts") -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / f"train_run_{timestamp}"
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(run_dir / "training.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
