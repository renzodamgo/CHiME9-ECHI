import hydra
from omegaconf import DictConfig
from pathlib import Path
import soundfile as sf
import soxr
from tqdm import tqdm

from src.shared.core_utils import get_session_tuples


def resample_for_enhancement(cfg: DictConfig):
    resample_all_rainbows(cfg)
    resample_all_sessions(cfg)


def resample_all_sessions(cfg: DictConfig):

    session_tuples = get_session_tuples(cfg.sessions_file, cfg.device, cfg.dataset)

    device_input_file = cfg.device_input_file
    device_output_file = cfg.device_output_file

    output_sample_rate = cfg.output_sample_rate

    session_devices = list(
        set([(session, device) for session, device, _ in session_tuples])
    )

    for session, device in tqdm(session_devices):
        dataset = session.split("_")[0]

        load_file = device_input_file.format(
            dataset=dataset, session=session, device=device
        )
        save_file = Path(
            device_output_file.format(dataset=dataset, session=session, device=device)
        )

        if save_file.exists():
            continue

        with open(load_file, "rb") as file:
            audio, fs = sf.read(file)

        if fs != output_sample_rate:
            audio = soxr.resample(audio, fs, output_sample_rate)

        if not save_file.parent.exists():
            save_file.parent.mkdir(parents=True)

        with open(save_file, "wb") as file:
            sf.write(file, audio, output_sample_rate)


def resample_all_rainbows(cfg: DictConfig):

    session_tuples = get_session_tuples(cfg.sessions_file, cfg.device, cfg.dataset)

    rainbow_orig_file = cfg.rainbow_input_file
    rainbow_new_file = cfg.rainbow_output_file

    output_sample_rate = cfg.output_sample_rate

    for session, _, pid in tqdm(session_tuples):
        dataset = session.split("_")[0]

        load_file = rainbow_orig_file.format(dataset=dataset, pid=pid)
        save_file = Path(rainbow_new_file.format(dataset=dataset, pid=pid))

        if save_file.exists():
            continue

        with open(load_file, "rb") as file:
            audio, fs = sf.read(file)

        if fs != output_sample_rate:
            audio = soxr.resample(audio, fs, output_sample_rate)

        if not save_file.parent.exists():
            save_file.parent.mkdir(parents=True)

        with open(save_file, "wb") as file:
            sf.write(file, audio, output_sample_rate)


@hydra.main(
    version_base=None, config_path="../../config/enhancement", config_name="main"
)
def main(cfg: DictConfig):
    print(cfg)
    resample_for_enhancement(cfg.resample)


if __name__ == "__main__":
    main()
