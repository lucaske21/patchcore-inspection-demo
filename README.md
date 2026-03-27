# PatchCore MVTec AD (Single-GPU)

## Introduction
本项目是一个基于 PatchCore 的 MVTec AD 异常检测简化实现，使用 PyTorch 单卡训练，并通过 YAML 配置进行参数管理。项目支持 MemoryBank 构建、异常分数计算、图像级/像素级 AUROC、PRO、F1 等指标评估，并自动生成混淆矩阵图（Ultralytics 风格），提供异常热图可视化输出，便于快速复现实验与进行工程化验证。适合学习 PatchCore 原理、快速基线实验与异常检测原型开发。


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
- F1 曲线图（自动保存）
- 混淆矩阵图（自动保存）

### F1 曲线

评估结束后自动生成 F1-Confidence 曲线图，保存至：
```
outputs/<category>/f1_curve.png
```

图像风格参考 Ultralytics，包含：
- 深色背景，同时展示 **图像级** 与 **像素级** 两条曲线（蓝色 / 粉色）
- X 轴为归一化置信度（0→1），Y 轴为 F1 分数
- 每条曲线在峰值处标注圆点、垂直虚线与 `F1=x.xxxx` 标注框
- 图例显示每条曲线的最优 F1 及对应置信度

### 混淆矩阵
评估结束后自动生成混淆矩阵图，保存至：
```
outputs/<category>/confusion_matrix.png
```

图像风格参考 Ultralytics，包含：
- 深色背景热力图，颜色深浅表示归一化比例（按行归一化，即 Recall 视角）
- 每格显示 **原始样本数** 与 **百分比**
- 底部 Metrics 面板，展示以下 8 项指标：

| 指标 | 说明 |
|---|---|
| Accuracy | 整体准确率 |
| Precision | 查准率（异常预测中真正异常的比例）|
| Recall | 查全率（真实异常被检出的比例）|
| Specificity | 特异度（正常样本正确判为正常的比例）|
| F1 | 最优 F1 分数（对应最优阈值）|
| Image AUROC | 图像级 ROC 曲线面积 |
| Pixel AUROC | 像素级 ROC 曲线面积 |
| PRO | Per-Region Overlap（FPR ≤ 0.3）|

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