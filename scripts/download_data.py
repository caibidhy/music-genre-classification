# scripts/download_data.py
import pathlib, sys
def main():
    root = pathlib.Path("data/gtzan")
    if root.exists():
        print(f"[OK] Found dataset at {root.resolve()}")
        return
    print("[INFO] 未检测到 data/gtzan。GTZAN 需从 Kaggle 手动下载：")
    print("  1) https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre")
    print("  2) 解压后将各流派文件夹放到 data/gtzan/<genre>/*.wav")
    sys.exit(1)
if __name__ == "__main__":
    main()
