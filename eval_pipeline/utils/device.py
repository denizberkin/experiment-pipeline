from typing import Any


def torch_load_device(device: Any) -> str | None:
    if device is None:
        return None
    if isinstance(device, str):
        return device
    if isinstance(device, list) and device:
        first = device[0]
        if isinstance(first, int):
            return f"cuda:{first}"
        if isinstance(first, str):
            return first
    return None
