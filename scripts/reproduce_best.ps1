python scripts\train_model.py --config configs\exp_best.yaml

python scripts\evaluate_model.py `
  --ckpt runs\exp_best_m96f512_B\best.pt `
  --val_csv data\index\val.csv `
  --frames 512
