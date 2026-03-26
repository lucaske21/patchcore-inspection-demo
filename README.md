# PatchCore MVTec AD (Single-GPU)

## Introduction
本项目是一个基于 PatchCore 的 MVTec AD 异常检测简化实现，使用 PyTorch 单卡训练，并通过 YAML 配置进行参数管理。项目支持 MemoryBank 构建、异常分数计算、图像级/像素级 AUROC、PRO、F1 等指标评估，并提供异常热图可视化输出，便于快速复现实验与进行工程化验证。适合学习 PatchCore 原理、快速基线实验与异常检测原型开发。


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