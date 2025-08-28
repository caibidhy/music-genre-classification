# scripts/organize_from_kaggle.py
from pathlib import Path
import shutil
import argparse

def find_genres_root(search_root: Path) -> Path:
    # 在解压目录下自动搜 "genres_original"（兼容 Data/genres_original 等各种层级）
    cands = list(search_root.rglob("genres_original"))
    if not cands:
        raise RuntimeError(f"Cannot find 'genres_original' under: {search_root}")
    # 优先选择目录而不是文件，取最短路径（最可能是正确那层）
    cands = [p for p in cands if p.is_dir()]
    cands.sort(key=lambda p: len(p.as_posix()))
    return cands[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", type=str, default="data/_kaggle",
                    help="Folder where you unzipped the Kaggle dataset")
    ap.add_argument("--dst", type=str, default="data/gtzan/genres",
                    help="Output folder in our expected structure")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    if not src_root.exists():
        raise RuntimeError(f"Source root not found: {src_root}")

    genres_root = find_genres_root(src_root)
    print(f"[INFO] Detected genres root: {genres_root}")

    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    n_total = 0
    for genre_dir in sorted(genres_root.iterdir()):
        if not genre_dir.is_dir():
            continue
        genre = genre_dir.name.lower()
        out_dir = dst_root / genre
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for wav in genre_dir.glob("*.wav"):
            shutil.copy2(wav, out_dir / wav.name)
            n += 1
        print(f"[COPY] {genre:10s}: {n} files")
        n_total += n

    print(f"\nDone. Copied {n_total} files into {dst_root}")

if __name__ == "__main__":
    main()

