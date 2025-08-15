import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
import csv
import torch
from typing import List
from pathlib import Path
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


class ECHIJoint(ECHI):
    """
    Yields:
      noisy:       [C, Tw] (same as your current 'noisy' file)
      target_all:  [K, Tw]
      spkid_all:   [K, Tr]  (Rainbow enrollments per PID)
      id:          str
      fs:          int
    """

    def make_manifest(self):
        self.manifest = []
        end = False

        for meta in self.metadata:
            try:
                device_pos = int(meta[f"{self.audio_device}_pos"])
            except ValueError:
                continue

            # PIDs seated at the other 3 positions
            pids = [meta[f"pos{i}"] for i in range(1, 5) if i != device_pos]

            # Load segments CSVs for each PID and intersect by index
            seg_lists = []
            for pid in pids:
                seg_csv = self.segments_file.format(
                    dataset=self.subset,
                    session=meta["session"],
                    device=self.audio_device,
                    pid=pid,
                )
                if not Path(seg_csv).exists():
                    seg_lists = []  # force skip
                    break
                with open(seg_csv, "r") as f:
                    segs = list(csv.DictReader(f, fieldnames=["index", "start", "end"]))
                seg_lists.append({int(s["index"]): s for s in segs})

            if len(seg_lists) != len(pids):
                continue

            common_idxs = set(seg_lists[0].keys())
            for d in seg_lists[1:]:
                common_idxs &= set(d.keys())
            if not common_idxs:
                continue

            for idx in sorted(common_idxs):
                # Build file paths for this segment idx across all PIDs
                seg_ok = True
                entry = {
                    "id": f'{meta["session"]}_{self.audio_device}_seg{idx:03d}',
                    "session": meta["session"],
                    "device": self.audio_device,
                    "idx": idx,
                    "pids": pids,
                    "noisy": None,  # will take from first pid (same audio content)
                    "target_all": [],  # per-pid segment
                    "spkid_all": [],  # per-pid enrollment (full rainbow file)
                }

                for j, pid in enumerate(pids):
                    noisy_path = self.signal_paths["noisy"].format(
                        dataset=self.subset,
                        session=meta["session"],
                        device=self.audio_device,
                        pid=pid,
                        segment=str(idx).zfill(3),
                    )
                    ref_path = self.signal_paths["target"].format(
                        dataset=self.subset,
                        session=meta["session"],
                        device=self.audio_device,
                        pid=pid,
                        segment=str(idx).zfill(3),
                    )
                    spk_path = self.signal_paths["spkid"].format(
                        dataset=self.subset, pid=pid
                    )  # Rainbow is not segmented

                    # All three must exist
                    if not (
                        Path(noisy_path).exists()
                        and Path(ref_path).exists()
                        and Path(spk_path).exists()
                    ):
                        seg_ok = False
                        break

                    if entry["noisy"] is None:
                        entry["noisy"] = (
                            noisy_path  # any pid's noisy has the same mixture
                        )

                    entry["target_all"].append(ref_path)
                    entry["spkid_all"].append(spk_path)

                if seg_ok:
                    self.manifest.append(entry)

                if self.debug and len(self.manifest) >= 10:
                    end = True
                    break
            if end:
                break

    def __getitem__(self, index):
        meta = self.manifest[index]
        out = {"id": meta["id"]}

        # noisy (C, Tw)
        noisy, nfs = torchaudio.load(str(meta["noisy"]))

        # K targets (1, Tw) each -> stack to [K, Tw]
        targets = []
        tfs_set = set()
        for p in meta["target_all"]:
            t, tfs = torchaudio.load(str(p))
            t = t.squeeze(0)
            targets.append(t)
            tfs_set.add(tfs)
        assert len(tfs_set) == 1, f"Inconsistent fs for targets: {tfs_set}"
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)  # [K, Tw]
        tfs = tfs_set.pop()

        # K enrollments (Rainbow), variable length -> [K, Tr]
        spks = []
        sfs_set = set()
        for p in meta["spkid_all"]:
            s, sfs = torchaudio.load(str(p))
            s = s.squeeze(0)
            spks.append(s)
            sfs_set.add(sfs)
        assert len(sfs_set) == 1, f"Inconsistent fs for spkids: {sfs_set}"
        spks = torch.nn.utils.rnn.pad_sequence(spks, batch_first=True)  # [K, Tr]
        sfs = sfs_set.pop()

        assert nfs == tfs == sfs, f"Sampling rate mismatch: {nfs}, {tfs}, {sfs}"

        out["noisy"] = noisy
        out["target_all"] = targets
        out["spkid_all"] = spks
        out["fs"] = nfs
        return out


def collate_fn_joint(batch: list[dict]):
    """
    Pads to:
      noisy         -> [B, C, Tw],    noisy_lens: [B]
      target_all    -> [B, K, Tw],    target_lens_all: [B, K]
      spkid_all     -> [B, K, Tr],    spkid_lens_all:  [B, K]
    Plus: id (list[str]) and fs (int)
    """
    ids = [x["id"] for x in batch]
    fs = batch[0]["fs"]

    # noisy
    from shared.signal_utils import combine_audio_list

    noisy = [x["noisy"] for x in batch]
    noisy_padded, noisy_lens = combine_audio_list(noisy)  # [B, C, Tw], [B]

    # K may vary between batches if something went wrong; enforce constant K
    K = batch[0]["target_all"].size(0)
    assert all(
        x["target_all"].size(0) == K for x in batch
    ), "Inconsistent K across batch"

    # targets
    max_Tw = max(x["target_all"].size(1) for x in batch)
    target_all = torch.zeros(len(batch), K, max_Tw)
    target_lens_all = torch.zeros(len(batch), K, dtype=torch.long)
    for b, x in enumerate(batch):
        Tw = x["target_all"].size(1)
        target_all[b, :, :Tw] = x["target_all"]
        target_lens_all[b] = Tw

    # enrollments
    max_Tr = max(x["spkid_all"].size(1) for x in batch)
    spkid_all = torch.zeros(len(batch), K, max_Tr)
    spkid_lens_all = torch.zeros(len(batch), K, dtype=torch.long)
    for b, x in enumerate(batch):
        Tr = x["spkid_all"].size(1)
        spkid_all[b, :, :Tr] = x["spkid_all"]
        spkid_lens_all[b] = Tr

    return {
        "id": ids,
        "fs": fs,
        "noisy": noisy_padded,
        "noisy_lens": noisy_lens,
        "target_all": target_all,
        "target_lens_all": target_lens_all,
        "spkid_all": spkid_all,
        "spkid_lens_all": spkid_lens_all,
    }
