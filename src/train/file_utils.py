import json
import soundfile as sf
from pathlib import Path


def read_json(path: Path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def write_json(path: Path, data: list[dict]):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def get_audio_len(path: Path, unit: str):
    """
    Calculates the duration (in seconds) of an audio file.
    Args:
        path (Path): The path to the audio file.
        unit (str): The unit of the duration (seconds, samples)
    Returns:
        float: The length of the audio file in seconds.
    Raises:
        RuntimeError: If the file cannot be opened or read as an audio file.
    """
    if not path.exists():
        raise RuntimeError(f"File path does not exist:\n{str(path)}")
    with sf.SoundFile(str(path)) as file:
        length = file.frames
        if unit == "seconds":
            length /= file.samplerate
    return length
