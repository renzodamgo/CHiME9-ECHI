from typing import Callable
from pathlib import Path


def get_enhance_fn(exp_dir: Path, device: str) -> tuple[Callable, dict]:

    name = exp_dir.name

    if name == "passthrough":
        from inference.passthrough import process_session

        return process_session, {"target_sr": 16000}
    elif name == "baseline":
        from inference.baseline import get_process

        return get_process(exp_dir, device)

    raise ValueError(f"Enhance option {name} not recognised. Add code here!")
