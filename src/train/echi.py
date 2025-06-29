import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
import csv

from shared.signal_utils import combine_audio_list


from typing import Any


def collate_fn(batch: list[dict[str, Any]]):
    new_out: dict[str, Any] = {"id": [x["id"] for x in batch], "fs": batch[0]["fs"]}

    for audio_type in ["noisy", "target", "spkid"]:
        audio = [x[audio_type] for x in batch]
        audio, lens = combine_audio_list(audio)
        new_out[audio_type] = audio
        new_out[audio_type + "_lens"] = lens

    return new_out


class ECHI(Dataset):
    def __init__(
        self,
        subset: str,
        audio_device: str,
        noisy_signal: str,
        ref_signal: str,
        rainbow_signal: str,
        sessions_file: str,
        segments_file: str,
        debug: bool,
    ):
        super().__init__()
        self.subset = subset
        self.audio_device = audio_device

        with open(sessions_file.format(dataset=subset), "r") as f:
            self.metadata = list(csv.DictReader(f))

        self.segments_file = segments_file

        self.signal_paths = {
            "noisy": noisy_signal,
            "target": ref_signal,
            "spkid": rainbow_signal,
        }

        self.segment_samples = 16000 * 4

        self.debug = debug

        self.manifest: list[dict]
        self.make_manifest()

    def make_manifest(self):
        self.manifest = []

        end = False

        for meta in self.metadata:

            try:
                device_pos = int(meta[f"{self.audio_device}_pos"])
            except ValueError:
                continue
            pids = [meta[f"pos{i}"] for i in range(1, 5) if i != device_pos]

            for pid in pids:
                with open(
                    self.segments_file.format(
                        dataset=self.subset,
                        session=meta["session"],
                        device=self.audio_device,
                        pid=pid,
                    ),
                    "r",
                ) as f:
                    segments = list(
                        csv.DictReader(f, fieldnames=["index", "start", "end"])
                    )

                self.manifest += self.get_segment_paths(meta["session"], pid, segments)

                if self.debug and len(self.manifest) > 10:
                    self.manifest = self.manifest[:10]
                    end = True
                    break
            if end:
                break

    def get_segment_paths(self, session, pid, segments) -> list[dict]:

        good_files = []
        for seg in segments:
            all_good = True
            seg_fpaths = {}
            for audio_type, fpath in self.signal_paths.items():
                this_fpath = fpath.format(
                    dataset=self.subset,
                    session=session,
                    device=self.audio_device,
                    pid=pid,
                    segment=str(seg["index"]).zfill(3),
                )

                if not Path(this_fpath).exists():
                    all_good = False
                    break

                seg_fpaths[audio_type] = this_fpath
            if all_good:
                length = (int(seg["end"]) - int(seg["start"])) / 16000
                if length < 1:
                    continue

                seg_fpaths["id"] = seg_fpaths["noisy"].split("/")[-1][:-4]
                good_files.append(seg_fpaths)

        return good_files

    def __getitem__(self, index):
        meta = self.manifest[index]

        out = {"id": meta["id"]}

        noisy, nfs = torchaudio.load(str(meta["noisy"]))
        target, tfs = torchaudio.load(str(meta["target"]))
        spkid, sfs = torchaudio.load(str(meta["spkid"]))

        assert nfs == tfs, f"Noisy fs ({nfs}Hz) doesn't match Target fs ({tfs}Hz)"
        assert nfs == sfs, f"Noisy fs ({nfs}Hz) doesn't match SpkID fs ({sfs}Hz)"

        if noisy.shape[-1] > self.segment_samples:
            noisy = noisy[..., : self.segment_samples]
            target = target[..., : self.segment_samples]

        out["noisy"] = noisy
        out["target"] = target.squeeze(0)
        out["spkid"] = spkid.squeeze(0)
        out["fs"] = nfs

        return out

    def __len__(self):
        return len(self.manifest)
