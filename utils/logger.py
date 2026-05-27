import logging
from datetime import datetime
from pathlib import Path


def create_run_dir(base_dir: str = "training_artifacts") -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / f"train_run_{timestamp}"
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    return run_dir


class _ElapsedFormatter(logging.Formatter):
    """Formatter that prepends elapsed time and expected remaining time to every line.

    Preamble fields (space-separated, before [LEVEL]):
        2026-05-26 09:36:35   ← wall-clock (asctime)
        +00:00:43             ← elapsed since run start
        -01:02:17             ← ETA (remaining); '--:--:--' until first update
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start = datetime.now()
        self._eta_seconds: float | None = None   # None = not yet known

    def set_eta(self, remaining_seconds: float) -> None:
        """Called by the trainer after each timed eval block."""
        self._eta_seconds = max(0.0, remaining_seconds)

    def format(self, record: logging.LogRecord) -> str:
        # Elapsed
        elapsed = datetime.now() - self._start
        total_secs = int(elapsed.total_seconds())
        eh, rem = divmod(total_secs, 3600)
        em, es = divmod(rem, 60)
        record.elapsed = f"+{eh:02d}:{em:02d}:{es:02d}"

        # ETA
        if self._eta_seconds is None:
            record.eta = "--:--:--"
        else:
            rh, rem = divmod(int(self._eta_seconds), 3600)
            rm, rs = divmod(rem, 60)
            record.eta = f"-{rh:02d}:{rm:02d}:{rs:02d}"

        return super().format(record)


def set_logger_eta(logger: logging.Logger, remaining_seconds: float) -> None:
    """Update the ETA field shown in every subsequent log line."""
    for handler in logger.handlers:
        if isinstance(handler.formatter, _ElapsedFormatter):
            handler.formatter.set_eta(remaining_seconds)


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.DEBUG)

    # Clear any handlers from a previous call (getLogger returns a singleton).
    logger.handlers.clear()

    formatter = _ElapsedFormatter(
        "%(asctime)s %(elapsed)s %(eta)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(run_dir / "training.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
