# scripts/evaluate_model.py
import argparse
import csv
import pathlib
import numpy as np
import librosa
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder


class TinyLinear(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def extract_meanstd(path: str, sr: int, n_mels: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    feat = np.concatenate([S_db.mean(axis=1), S_db.std(axis=1)], axis=0).astype(np.float32)
    return feat  # (2 * n_mels,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", type=str, default="data/index/val.csv")
    ap.add_argument("--ckpt", type=str, default="runs/exp_meanstd/model.pt")
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--sr", type=int, default=22050)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt.get("label_encoder")
    if classes is None:
        raise SystemExit("The checkpoint is missing the label_encoder. Please retrain using the train script in this project.")

    # 以 checkpoint 的 n_mels 为准，避免不一致
    n_mels = int(ckpt.get("n_mels", args.n_mels))
    in_dim = n_mels * 2  # mean+std

    model = TinyLinear(in_dim, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    le = LabelEncoder().fit(classes)

    tot, correct = 0, 0
    with open(args.val_csv, "r", encoding="utf-8") as fp:
        rdr = csv.DictReader(fp)
        for r in rdr:
            feat = extract_meanstd(r["path"], args.sr, n_mels)
            x = torch.from_numpy(feat).unsqueeze(0)  # (1, 2*n_mels)
            pred_idx = model(x).argmax(1).item()
            if pred_idx == le.transform([r["label"]])[0]:
                correct += 1
            tot += 1

    if tot == 0:
        print("The Val set is empty. Please run preprocess_data.py first to generate the index.")
    else:
        print(f"Eval accuracy: {correct}/{tot} = {correct/tot:.3f}")


if __name__ == "__main__":
    main()
