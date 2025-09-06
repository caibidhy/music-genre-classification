# scripts/generate_predictions.py
import argparse, pathlib, numpy as np, librosa, torch, torch.nn as nn, csv

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
    ap.add_argument("--audio_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="runs/smoke_1ep/model.pt")
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--out_csv", type=str, default="predictions.csv")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt["label_encoder"]; n_classes = len(classes)
    model = TinyLinear(args.n_mels, n_classes); model.load_state_dict(ckpt["state_dict"]); model.eval()

    rows = []
    for wav in sorted(pathlib.Path(args.audio_dir).glob("*.wav")):
        y, _ = librosa.load(wav, sr=args.sr, mono=True)
        S = librosa.feature.melspectrogram(y=y, sr=args.sr, n_mels=args.n_mels, power=2.0)
        feat = librosa.power_to_db(S, ref=np.max).mean(axis=1).astype(np.float32)
        x = torch.from_numpy(feat).unsqueeze(0)
        pred = model(x).argmax(1).item()
        rows.append((str(wav), classes[pred]))

    with open(args.out_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp); w.writerow(["path","pred"])
        w.writerows(rows)
    print(f"Saved predictions to {args.out_csv}")

if __name__ == "__main__":
    main()
