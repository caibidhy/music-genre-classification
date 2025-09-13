# scripts/evaluate_model.py
import argparse, csv, numpy as np, librosa, torch, torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

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
    # 与 train_model.py 完全一致（无 Dropout）
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
    ap.add_argument("--ckpt", type=str, default="runs/exp_best_m96f512_B/best.pt")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mels", type=int, default=96)
    ap.add_argument("--frames", type=int, default=512)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop_length", type=int, default=512)
    args = ap.parse_args()

    # 安全加载（兼容旧版）
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location="cpu")

    classes = ckpt["label_encoder"]
    n_mels  = int(ckpt.get("n_mels", args.n_mels))
    frames  = int(ckpt.get("frames", args.frames))

    model = TinyCNN(len(classes))
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    le = LabelEncoder().fit(classes)

    y_true_idx, y_pred_idx = [], []
    pred_rows = []

    with open(args.val_csv, "r", encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            wav, true_lbl = r["path"], r["label"]
            y, _ = librosa.load(wav, sr=args.sr, mono=True)
            S_db = to_log_mel(y, args.sr, n_mels, args.n_fft, args.hop_length)
            S_db = center_crop_or_pad(S_db, frames)
            mu, sigma = S_db.mean(), S_db.std() + 1e-6
            S_db = (S_db - mu) / sigma
            x = torch.from_numpy(S_db).unsqueeze(0).unsqueeze(0).float()
            pred_idx = model(x).argmax(1).item()
            true_idx = le.transform([true_lbl])[0]

            y_true_idx.append(true_idx)
            y_pred_idx.append(pred_idx)
            pred_rows.append({"path": wav, "true": true_lbl, "pred": classes[pred_idx],
                              "correct": int(pred_idx == true_idx)})

    tot = len(y_true_idx)
    acc = float(np.mean(np.array(y_true_idx) == np.array(y_pred_idx))) if tot else 0.0
    print(f"Eval accuracy: {int(acc*tot)}/{tot} = {acc:.3f}")

    out_dir = Path(args.ckpt).parent; out_dir.mkdir(parents=True, exist_ok=True)
    # 明细
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path","true","pred","correct"])
        w.writeheader(); w.writerows(pred_rows)
    # 混淆矩阵 + 每类准确率
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(classes))))
    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow([""] + list(classes))
        for i,row in enumerate(cm):
            w.writerow([classes[i]] + list(map(int,row)))
    per_class_rows = []
    for i,c in enumerate(classes):
        support = int(cm[i].sum()); correct_i = int(cm[i, i])
        acc_i = correct_i / support if support else 0.0
        per_class_rows.append([c, correct_i, support, f"{acc_i:.3f}"])
    with open(out_dir / "per_class_accuracy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["class","correct","support","accuracy"])
        w.writerows(per_class_rows)
    # 文本报告
    (out_dir / "report.txt").write_text(
        classification_report(y_true_idx, y_pred_idx, target_names=classes, digits=3, zero_division=0),
        encoding="utf-8"
    )
    # 混淆矩阵图片（可选）
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
        plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
        plt.yticks(range(len(classes)), classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, int(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        fig.savefig(out_dir / "cm.png", dpi=150); plt.close(fig)
        print(f"Saved confusion matrix to {out_dir/'cm.png'}")
    except Exception as e:
        print(f"Skip CM plot: {e}")

if __name__ == "__main__":
    main()



