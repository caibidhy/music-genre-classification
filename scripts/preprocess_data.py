# scripts/preprocess_data.py
import argparse, csv, pathlib, random

GENRES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/gtzan/genres",
                    help="目录形如 data/gtzan/genres/<genre>/*.wav 或 *.au")
    ap.add_argument("--out_dir", type=str, default="data/index")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mini_per_class", type=int, default=0,
                    help="每类最多取N个样本；0=全量")
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rows = []; total = 0
    for g in GENRES:
        gdir = data_dir / g
        files = []
        for ext in ("*.wav","*.au"):
            files.extend(sorted(gdir.glob(ext)))
        if args.mini_per_class > 0:
            files = files[:args.mini_per_class]
        print(f"[{g}] found {len(files)} files under {gdir}")
        rows += [(str(f), g) for f in files]
        total += len(files)

    if total == 0:
        raise SystemExit(f"[ERROR] 没在 {data_dir} 找到音频，请检查路径/后缀。")

    rng.shuffle(rows)
    n_val = int(len(rows) * args.val_ratio)
    val_rows = rows[:n_val]; train_rows = rows[n_val:]

    for name, subset in [("train.csv", train_rows), ("val.csv", val_rows)]:
        with open(out_dir / name, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp); w.writerow(["path","label"]); w.writerows(subset)

    print(f"Saved: {out_dir/'train.csv'}  {out_dir/'val.csv'} "
          f"(total={len(rows)} train={len(train_rows)} val={len(val_rows)})")

if __name__ == "__main__":
    main()


