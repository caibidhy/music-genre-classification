# scripts/evaluate_model.py
import argparse, csv, numpy as np, librosa, torch, torch.nn as nn
from sklearn.preprocessing import LabelEncoder

def to_log_mel(y, sr, n_mels, n_fft, hop):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop, n_mels=n_mels, power=2.0)
    return librosa.power_to_db(S, ref=np.max)

def center_crop_or_pad(S, frames):
    T = S.shape[1]
    if T == frames: return S
    if T > frames:
        s = (T - frames) // 2
        return S[:, s:s+frames]
    pad = frames - T
    l = pad // 2; r = pad - l
    return np.pad(S, ((0,0),(l,r)), mode="reflect")

class TinyCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.cls = nn.Linear(64, n_classes)
    def forward(self,x):
        h = self.feat(x).view(x.size(0), -1)
        return self.cls(h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", type=str, default="data/index/val.csv")
    ap.add_argument("--ckpt", type=str, default="runs/exp_cnn/model.pt")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop_length", type=int, default=512)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt["label_encoder"]
    n_mels  = int(ckpt.get("n_mels", args.n_mels))
    frames  = int(ckpt.get("frames", args.frames))

    model = TinyCNN(len(classes))
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    le = LabelEncoder().fit(classes)

    tot=0; correct=0
    with open(args.val_csv, "r", encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            y, _ = librosa.load(r["path"], sr=args.sr, mono=True)
            S_db = to_log_mel(y, args.sr, n_mels, args.n_fft, args.hop_length)
            S_db = center_crop_or_pad(S_db, frames)
            mu, sigma = S_db.mean(), S_db.std() + 1e-6
            S_db = (S_db - mu) / sigma
            x = torch.from_numpy(S_db).unsqueeze(0).unsqueeze(0).float()  # (1,1,n_mels,frames)
            pred = model(x).argmax(1).item()
            if pred == le.transform([r["label"]])[0]:
                correct += 1
            tot += 1
    if tot==0:
        print("Val is empty, please run preprocess_data.py first.")
    else:
        print(f"Eval accuracy: {correct}/{tot} = {correct/tot:.3f}")

if __name__ == "__main__":
    main()

