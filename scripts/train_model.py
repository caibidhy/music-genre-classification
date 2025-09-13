# scripts/train_model.py
import argparse, csv as pycsv, pathlib, random
import numpy as np, librosa, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# ---------------- Utils ----------------
def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def to_log_mel(y, sr, n_mels, n_fft, hop):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop, n_mels=n_mels, power=2.0)
    return librosa.power_to_db(S, ref=np.max)  # (n_mels, T)

def center_crop_or_pad(S, frames):
    T = S.shape[1]
    if T == frames: return S
    if T > frames:
        s = (T - frames) // 2
        return S[:, s:s+frames]
    pad = frames - T
    l = pad // 2; r = pad - l
    return np.pad(S, ((0,0),(l,r)), mode="reflect")

# ----- waveform augment -----
def augment_waveform(y, sr, p=0.4, gain_db=2.0, shift_ms=30, noise_snr_db=35.0):
    if np.random.rand() > p: return y
    # gain
    g_db = (np.random.rand()*2 - 1.0) * gain_db
    y = y * (10.0 ** (g_db / 20.0))
    # shift
    kmax = int(sr * shift_ms / 1000.0)
    if kmax > 0:
        k = np.random.randint(-kmax, kmax+1)
        if k != 0:
            y = np.roll(y, k)
            if k > 0: y[:k] = 0
            else:     y[k:] = 0
    # noise
    sig_pow = float(np.mean(y.astype(np.float64)**2))
    if sig_pow > 1e-12 and noise_snr_db is not None:
        snr_lin = 10.0 ** (noise_snr_db / 10.0)
        noise_pow = sig_pow / snr_lin
        y = y + np.random.randn(*y.shape) * np.sqrt(noise_pow)
    return y

# ----- spec augment -----
def spec_augment(S_db, num_time_masks=1, num_freq_masks=1, time_max=8, freq_max=4):
    S = S_db.copy()
    n_mels, T = S.shape
    for _ in range(num_freq_masks):
        f = np.random.randint(0, max(1, min(freq_max, n_mels)))
        f0 = np.random.randint(0, max(1, n_mels - f))
        S[f0:f0+f, :] = 0.0
    for _ in range(num_time_masks):
        t = np.random.randint(0, max(1, min(time_max, T)))
        t0 = np.random.randint(0, max(1, T - t))
        S[:, t0:t0+t] = 0.0
    return S

# ---------------- Dataset ----------------
class MelSpecDataset(Dataset):
    def __init__(self, csv_path, n_mels=96, frames=512, sr=22050, n_fft=2048,
                 hop_length=512, normalize=True, is_train=False,
                 aug_prob=0.4, aug_gain_db=2.0, aug_shift_ms=30, aug_noise_snr=35.0,
                 spec_time_mask=1, spec_freq_mask=1, spec_time_max=8, spec_freq_max=4):
        import csv
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
        self.is_train = is_train

        self.aug_prob = aug_prob
        self.aug_gain_db = aug_gain_db
        self.aug_shift_ms = aug_shift_ms
        self.aug_noise_snr = aug_noise_snr
        self.spec_time_mask = spec_time_mask
        self.spec_freq_mask = spec_freq_mask
        self.spec_time_max = spec_time_max
        self.spec_freq_max = spec_freq_max

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        y, sr = librosa.load(path, sr=self.sr, mono=True)
        if self.is_train:
            y = augment_waveform(y, sr, p=self.aug_prob, gain_db=self.aug_gain_db,
                                 shift_ms=self.aug_shift_ms, noise_snr_db=self.aug_noise_snr)
        S_db = to_log_mel(y, sr, self.n_mels, self.n_fft, self.hop)
        S_db = center_crop_or_pad(S_db, self.frames)
        if self.is_train and (self.spec_time_mask>0 or self.spec_freq_mask>0):
            S_db = spec_augment(S_db, self.spec_time_mask, self.spec_freq_mask,
                                self.spec_time_max, self.spec_freq_max)
        if self.normalize:
            mu, sigma = S_db.mean(), S_db.std() + 1e-6
            S_db = (S_db - mu) / sigma
        x = torch.from_numpy(S_db).unsqueeze(0).float()
        y_idx = torch.tensor(self.le.transform([label])[0], dtype=torch.long)
        return x, y_idx

# ---------------- Model ----------------
class TinyCNN(nn.Module):
    """与 evaluate_model.py 完全一致（无 Dropout 的最佳版）"""
    def __init__(self, n_classes: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.cls = nn.Linear(64, n_classes)
    def forward(self, x):
        h = self.feat(x).view(x.size(0), -1)
        return self.cls(h)

# ---------------- Logging ----------------
def write_log_row(csv_path, row_dict, header):
    file_exists = pathlib.Path(csv_path).exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = pycsv.DictWriter(f, fieldnames=header)
        if not file_exists: w.writeheader()
        w.writerow(row_dict)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="", help="YAML 配置路径（命令行可覆盖）")

    # data
    ap.add_argument("--train_csv", type=str, default="data/index/train.csv")
    ap.add_argument("--val_csv",   type=str, default="data/index/val.csv")

    # features
    ap.add_argument("--n_mels", type=int, default=96)
    ap.add_argument("--frames", type=int, default=512)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop_length", type=int, default=512)

    # training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output_dir", type=str, default="runs/exp_best_m96f512_B")
    ap.add_argument("--seed", type=int, default=42)

    # early stop + scheduler
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min_delta", type=float, default=0.005)
    ap.add_argument("--lr_patience", type=int, default=2)
    ap.add_argument("--lr_factor", type=float, default=0.5)

    # augment
    ap.add_argument("--augment_prob", type=float, default=0.4)
    ap.add_argument("--aug_gain_db", type=float, default=2.0)
    ap.add_argument("--aug_shift_ms", type=int, default=30)
    ap.add_argument("--aug_noise_snr", type=float, default=35.0)
    ap.add_argument("--spec_time_mask", type=int, default=1)
    ap.add_argument("--spec_freq_mask", type=int, default=1)
    ap.add_argument("--spec_time_max", type=int, default=8)
    ap.add_argument("--spec_freq_max", type=int, default=4)

    # debug
    ap.add_argument("--limit_train_batches", type=int, default=0)
    ap.add_argument("--limit_val_batches", type=int, default=0)

    args = ap.parse_args()

    # ---- load YAML (if any) and let CLI override ----
    if args.config:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        def pick(*keys, default=None):
            d = cfg
            for k in keys:
                if d is None: return default
                d = d.get(k)
            return d if d is not None else default

        # data
        args.train_csv   = pick("data","train_csv",   default=args.train_csv)
        args.val_csv     = pick("data","val_csv",     default=args.val_csv)
        # features
        args.n_mels      = pick("features","n_mels",      default=args.n_mels)
        args.frames      = pick("features","frames",      default=args.frames)
        args.n_fft       = pick("features","n_fft",       default=args.n_fft)
        args.hop_length  = pick("features","hop_length",  default=args.hop_length)
        # train
        args.epochs      = pick("train","epochs",      default=args.epochs)
        args.batch_size  = pick("train","batch_size",  default=args.batch_size)
        args.num_workers = pick("train","num_workers", default=args.num_workers)
        args.device      = pick("train","device",      default=args.device)
        args.output_dir  = pick("train","output_dir",  default=args.output_dir)
        # early stop
        args.patience    = pick("earlystop","patience", default=args.patience)
        args.min_delta   = pick("earlystop","min_delta", default=args.min_delta)
        # LR scheduler
        args.lr_patience = pick("lr_scheduler","patience", default=args.lr_patience)
        args.lr_factor   = pick("lr_scheduler","factor",   default=args.lr_factor)
        # augment
        args.augment_prob   = pick("augment","augment_prob",   default=args.augment_prob)
        args.aug_gain_db    = pick("augment","aug_gain_db",    default=args.aug_gain_db)
        args.aug_shift_ms   = pick("augment","aug_shift_ms",   default=args.aug_shift_ms)
        args.aug_noise_snr  = pick("augment","aug_noise_snr",  default=args.aug_noise_snr)
        args.spec_time_mask = pick("augment","spec_time_mask", default=args.spec_time_mask)
        args.spec_freq_mask = pick("augment","spec_freq_mask", default=args.spec_freq_mask)
        args.spec_time_max  = pick("augment","spec_time_max",  default=args.spec_time_max)
        args.spec_freq_max  = pick("augment","spec_freq_max",  default=args.spec_freq_max)

    # ---- train ----
    set_global_seed(args.seed)
    out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    log_csv = out / "log.csv"

    train_set = MelSpecDataset(args.train_csv, n_mels=args.n_mels, frames=args.frames,
                               sr=22050, n_fft=args.n_fft, hop_length=args.hop_length,
                               normalize=True, is_train=True,
                               aug_prob=args.augment_prob, aug_gain_db=args.aug_gain_db,
                               aug_shift_ms=args.aug_shift_ms, aug_noise_snr=args.aug_noise_snr,
                               spec_time_mask=args.spec_time_mask, spec_freq_mask=args.spec_freq_mask,
                               spec_time_max=args.spec_time_max, spec_freq_max=args.spec_freq_max)

    val_set   = MelSpecDataset(args.val_csv,   n_mels=args.n_mels, frames=args.frames,
                               sr=22050, n_fft=args.n_fft, hop_length=args.hop_length,
                               normalize=True, is_train=False)

    n_classes = len(train_set.le.classes_)
    model = TinyCNN(n_classes).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce  = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=args.lr_patience, factor=args.lr_factor, verbose=True
    )

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=False)

    train_loader = make_loader(train_set, True)
    val_loader   = make_loader(val_set, False)

    best_acc = -1.0
    epochs_no_improve = 0
    header = ["epoch","train_loss","val_acc","lr"]

    global_step = 0
    for ep in range(args.epochs):
        model.train()
        running = 0.0; steps = 0
        for bi, (x, y) in enumerate(train_loader):
            if args.limit_train_batches and bi >= args.limit_train_batches: break
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward(); opt.step()
            running += loss.item(); steps += 1
            if bi % 10 == 0:
                print(f"epoch {ep} step {global_step} loss {loss.item():.4f}")
            global_step += 1
        train_loss = running / max(steps, 1)

        model.eval(); tot=0; correct=0
        with torch.no_grad():
            for bi, (x, y) in enumerate(val_loader):
                if args.limit_val_batches and bi >= args.limit_val_batches: break
                x, y = x.to(args.device), y.to(args.device)
                pred = model(x).argmax(1)
                correct += (pred==y).float().sum().item()
                tot += y.numel()
        val_acc = correct / max(tot, 1)
        lr_now = opt.param_groups[0]["lr"]
        print(f"[val] epoch {ep} acc={val_acc:.3f} on {tot} samples | lr={lr_now:.5f}")

        scheduler.step(val_acc)
        improved = val_acc > best_acc + args.min_delta
        if improved:
            best_acc = val_acc; epochs_no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "label_encoder": train_set.le.classes_.tolist(),
                "n_mels": args.n_mels, "frames": args.frames,
                "feature_type": "logmel_2d_norm_centercrop+light_aug",
            }, out / "best.pt")
            print(f"  ✔ New best acc={best_acc:.3f}. Saved best to {out/'best.pt'}")
        else:
            epochs_no_improve += 1
            print(f"  ↳ No improvement ({epochs_no_improve}/{args.patience})")

        write_log_row(log_csv, {"epoch": ep, "train_loss": f"{train_loss:.4f}",
                                "val_acc": f"{val_acc:.4f}", "lr": f"{lr_now:.6f}"}, header)

        if epochs_no_improve >= args.patience:
            print("Early stopping triggered."); break

    torch.save({
        "state_dict": model.state_dict(),
        "label_encoder": train_set.le.classes_.tolist(),
        "n_mels": args.n_mels, "frames": args.frames,
        "feature_type": "logmel_2d_norm_centercrop+light_aug",
    }, out / "model.pt")
    print(f"Saved last checkpoint to {out/'model.pt'}; best at {out/'best.pt'}")

if __name__ == "__main__":
    main()




