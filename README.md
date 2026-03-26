# PatchCore MVTec AD (Single-GPU)

## 1. 安装
```bash
pip install -r requirements.txt
```

## 2. 数据目录
```
data/mvtec_ad/
  bottle/
    train/good/*.png
    test/good/*.png
    test/broken_large/*.png
    ground_truth/broken_large/*.png
```

## 3. 训练
```bash
python scripts/train.py configs/patchcore_mvtec.yaml
```

## 4. 测试 + 指标
```bash
python scripts/eval.py configs/patchcore_mvtec.yaml
```

输出指标：
- Image AUROC
- Pixel AUROC
- Image F1（max-f1）
- Pixel F1（max-f1）
- PRO（FPR<=0.3）

## 5. 异常热图可视化
默认保存到：
```
outputs/<category>/visuals/
  xxx_heatmap.png
  xxx_overlay.png
```

可在 `configs/patchcore_mvtec.yaml` 中配置：
```yaml
eval:
  save_anomaly_maps: true
  overlay_alpha: 0.5
```