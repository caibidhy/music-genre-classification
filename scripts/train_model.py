# scripts/train_model.py
import argparse
import csv
import pathlib
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


class MelMeanStdDataset(Dataset):
    """
    读取 CSV（两列：path,label），为每条音频生成 log-mel 的 [mean || std] 特征。
    输出:
      x: FloatTensor, shape=(2*n_mels,)
      y: LongTensor,  标量类别索引
    """
    def __init__(
        self,
        csv_path: str,
        n_mels: int = 64,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        self.items = []
        with open(csv_path, "r", encoding="utf-8") as fp:
            rdr = csv.DictReader(fp)
            for r in rdr:
                self.items.append((r["path"], r["label"]))
        if not self.items:
            raise ValueError(f"{csv_path} 为空，没有样本。请先运行 preprocess_data.py 生成索引。")

        self.le = LabelEncoder().fit([lbl for _, lbl in self.items])
        self.n_mels = n_mels
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        y, sr = librosa.load(path, sr=self.sr, mono=True)

        # log-mel
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop,
            n_mels=self.n_mels,
            power=2.0,
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        # 关键改动：拼接 mean 和 std（沿时间轴）
        feat = np.concatenate(
            [S_db.mean(axis=1), S_db.std(axis=1)],
            axis=0,
        ).astype(np.float32)  # (2 * n_mels,)

        x = torch.from_numpy(feat)
        y_idx = torch.tensor(self.le.transform([label])[0], dtype=torch.long)
        return x, y_idx


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="data/index/train.csv")
    ap.add_argument("--val_csv", type=str, default="data/index/val.csv")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output_dir", type=str, default="runs/exp_meanstd")
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--limit_train_batches", type=int, default=0)
    ap.add_argument("--limit_val_batches", type=int, default=0)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = MelMeanStdDataset(args.train_csv, n_mels=args.n_mels)
    val_set = MelMeanStdDataset(args.val_csv, n_mels=args.n_mels)

    n_classes = len(train_set.le.classes_)
    in_dim = args.n_mels * 2  # mean+std
    model = TinyLinear(in_dim, n_classes).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=False,
        )

    train_loader = make_loader(train_set, True)
    val_loader = make_loader(val_set, False)

    global_step = 0
    for ep in range(args.epochs):
        # train
        model.train()
        for bi, (x, y) in enumerate(train_loader):
            if args.limit_train_batches and bi >= args.limit_train_batches:
                break
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            if bi % 10 == 0:
                print(f"epoch {ep} step {global_step} loss {loss.item():.4f}")
            global_step += 1

        # validate
        model.eval()
        tot, correct = 0, 0
        with torch.no_grad():
            for bi, (x, y) in enumerate(val_loader):
                if args.limit_val_batches and bi >= args.limit_val_batches:
                    break
                x, y = x.to(args.device), y.to(args.device)
                logits = model(x)
                pred = logits.argmax(1)
                correct += (pred == y).float().sum().item()
                tot += y.numel()
        if tot > 0:
            print(f"[val] epoch {ep} acc={correct/tot:.3f} on {tot} samples")

    # 保存权重 + 类别列表 + 特征配置
    torch.save(
        {
            "state_dict": model.state_dict(),
            "label_encoder": train_set.le.classes_.tolist(),
            "n_mels": args.n_mels,
            "feature_type": "mel_mean_std",
        },
        out_dir / "model.pt",
    )
    print(f"Saved checkpoint to {out_dir/'model.pt'}")


if __name__ == "__main__":
    main()


