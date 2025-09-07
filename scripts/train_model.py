# scripts/train_model.py
import argparse, csv, pathlib, numpy as np, librosa, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

def to_log_mel(y, sr, n_mels, n_fft, hop):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop, n_mels=n_mels, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db  # (n_mels, T)

def center_crop_or_pad(S, frames):
    # S: (n_mels, T) -> 固定到 frames
    T = S.shape[1]
    if T == frames:
        return S
    if T > frames:
        s = (T - frames) // 2
        return S[:, s:s+frames]
    pad = frames - T
    l = pad // 2; r = pad - l
    return np.pad(S, ((0,0),(l,r)), mode="reflect")

class MelSpecDataset(Dataset):
    def __init__(self, csv_path, n_mels=64, frames=256, sr=22050, n_fft=2048,
                 hop_length=512, normalize=True):
        self.items = []
        with open(csv_path, "r", encoding="utf-8") as fp:
            for r in csv.DictReader(fp):
                self.items.append((r["path"], r["label"]))
        if not self.items:
            raise ValueError(f"{csv_path} 为空，请先运行 preprocess_data.py。")

        self.le = LabelEncoder().fit([lbl for _, lbl in self.items])
        self.n_mels, self.frames = n_mels, frames
        self.sr, self.n_fft, self.hop = sr, n_fft, hop_length
        self.normalize = normalize

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        y, sr = librosa.load(path, sr=self.sr, mono=True)
        S_db = to_log_mel(y, sr, self.n_mels, self.n_fft, self.hop)
        S_db = center_crop_or_pad(S_db, self.frames)
        if self.normalize:
            mu, sigma = S_db.mean(), S_db.std() + 1e-6
            S_db = (S_db - mu) / sigma
        x = torch.from_numpy(S_db).unsqueeze(0).float()  # (1, n_mels, frames)
        y_idx = torch.tensor(self.le.transform([label])[0], dtype=torch.long)
        return x, y_idx

class TinyCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # /2
            nn.Conv2d(16,32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # /4
            nn.Conv2d(32,64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))                                                       # -> (B,64,1,1)
        )
        self.cls = nn.Linear(64, n_classes)

    def forward(self, x):
        h = self.feat(x)           # (B,64,1,1)
        h = h.view(x.size(0), -1)  # (B,64)
        return self.cls(h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="data/index/train.csv")
    ap.add_argument("--val_csv",   type=str, default="data/index/val.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output_dir", type=str, default="runs/exp_cnn")
    # 特征参数
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop_length", type=int, default=512)
    # 调试参数
    ap.add_argument("--limit_train_batches", type=int, default=0)
    ap.add_argument("--limit_val_batches",   type=int, default=0)
    args = ap.parse_args()

    out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    train_set = MelSpecDataset(args.train_csv, n_mels=args.n_mels, frames=args.frames,
                               sr=22050, n_fft=args.n_fft, hop_length=args.hop_length)
    val_set   = MelSpecDataset(args.val_csv,   n_mels=args.n_mels, frames=args.frames,
                               sr=22050, n_fft=args.n_fft, hop_length=args.hop_length)

    n_classes = len(train_set.le.classes_)
    model = TinyCNN(n_classes).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce  = nn.CrossEntropyLoss()

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=False)

    train_loader = make_loader(train_set, True)
    val_loader   = make_loader(val_set, False)

    global_step = 0
    for ep in range(args.epochs):
        model.train()
        for bi, (x, y) in enumerate(train_loader):
            if args.limit_train_batches and bi >= args.limit_train_batches: break
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward(); opt.step()
            if bi % 10 == 0:
                print(f"epoch {ep} step {global_step} loss {loss.item():.4f}")
            global_step += 1

        # val
        model.eval(); tot=0; correct=0
        with torch.no_grad():
            for bi, (x, y) in enumerate(val_loader):
                if args.limit_val_batches and bi >= args.limit_val_batches: break
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x).argmax(1)
                correct += (pred==y).float().sum().item()
                tot += y.numel()
        if tot>0:
            print(f"[val] epoch {ep} acc={correct/tot:.3f} on {tot} samples")

    torch.save({
        "state_dict": model.state_dict(),
        "label_encoder": train_set.le.classes_.tolist(),
        "n_mels": args.n_mels,
        "frames": args.frames,
        "feature_type": "logmel_2d_norm_centercrop",
    }, out / "model.pt")
    print(f"Saved checkpoint to {out/'model.pt'}")

if __name__ == "__main__":
    main()


