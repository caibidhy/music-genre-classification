# scripts/train_model.py
import argparse, csv, pathlib, numpy as np, librosa, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class MelMeanDataset(Dataset):
    def __init__(self, csv_path, n_mels=64, sr=22050, n_fft=2048, hop_length=512):
        self.items = []
        with open(csv_path, "r", encoding="utf-8") as fp:
            rdr = csv.DictReader(fp)
            for r in rdr: self.items.append((r["path"], r["label"]))
        if not self.items:
            raise ValueError(f"{csv_path} 为空，没有样本。请先运行 preprocess_data.py")
        self.le = LabelEncoder().fit([lbl for _, lbl in self.items])
        self.n_mels, self.sr, self.n_fft, self.hop = n_mels, sr, n_fft, hop_length

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        y, sr = librosa.load(path, sr=self.sr, mono=True)
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop,
            n_mels=self.n_mels, power=2.0
        )
        feat = librosa.power_to_db(S, ref=np.max).mean(axis=1).astype(np.float32)
        x = torch.from_numpy(feat)   # (n_mels,)
        y_lbl = torch.tensor(self.le.transform([label])[0], dtype=torch.long)
        return x, y_lbl

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
    ap.add_argument("--train_csv", type=str, default="data/index/train.csv")
    ap.add_argument("--val_csv", type=str, default="data/index/val.csv")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output_dir", type=str, default="runs/smoke_1ep")
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--limit_train_batches", type=int, default=0)
    ap.add_argument("--limit_val_batches", type=int, default=0)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_set = MelMeanDataset(args.train_csv, n_mels=args.n_mels)
    val_set   = MelMeanDataset(args.val_csv,   n_mels=args.n_mels)

    n_classes = len(train_set.le.classes_)
    model = TinyLinear(args.n_mels, n_classes).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
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

        # eval
        model.eval(); tot_acc, tot = 0.0, 0
        with torch.no_grad():
            for bi, (x, y) in enumerate(val_loader):
                if args.limit_val_batches and bi >= args.limit_val_batches: break
                x, y = x.to(args.device), y.to(args.device)
                logits = model(x)
                tot_acc += (logits.argmax(1) == y).float().sum().item()
                tot += y.numel()
        if tot > 0:
            print(f"[val] epoch {ep} acc={tot_acc/tot:.3f} on {tot} samples")

    torch.save({
        "state_dict": model.state_dict(),
        "label_encoder": train_set.le.classes_.tolist(),
        "n_mels": args.n_mels
    }, out_dir / "model.pt")
    print(f"Saved checkpoint to {out_dir/'model.pt'}")

if __name__ == "__main__":
    main()

