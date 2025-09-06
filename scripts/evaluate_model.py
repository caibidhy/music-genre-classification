# scripts/evaluate_model.py
import argparse, csv, pathlib, numpy as np, librosa, torch, torch.nn as nn
from sklearn.preprocessing import LabelEncoder

class TinyLinear(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    def forward(self, x): return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", type=str, default="data/index/val.csv")
    ap.add_argument("--ckpt", type=str, default="runs/smoke_1ep/model.pt")
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--sr", type=int, default=22050)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt["label_encoder"]; n_classes = len(classes)
    model = TinyLinear(args.n_mels, n_classes); model.load_state_dict(ckpt["state_dict"]); model.eval()

    # build label encoder to ints
    le = LabelEncoder().fit(classes)

    tot, correct = 0, 0
    with open(args.val_csv, "r", encoding="utf-8") as fp:
        rdr = csv.DictReader(fp)
        for r in rdr:
            y, _ = librosa.load(r["path"], sr=args.sr, mono=True)
            S = librosa.feature.melspectrogram(y=y, sr=args.sr, n_mels=args.n_mels, power=2.0)
            feat = librosa.power_to_db(S, ref=np.max).mean(axis=1).astype(np.float32)
            x = torch.from_numpy(feat).unsqueeze(0)  # (1, n_mels)
            pred = model(x).argmax(1).item()
            if pred == le.transform([r["label"]])[0]:
                correct += 1
            tot += 1
    print(f"Eval accuracy: {correct}/{tot} = {correct/tot:.3f}")

if __name__ == "__main__":
    main()
