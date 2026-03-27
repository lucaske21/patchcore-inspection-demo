import argparse
import json
import os
import platform
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from skimage.measure import label

from patchcore.metrics import plot_roc_curve, plot_f1_curve, plot_confusion_matrix


@dataclass
class EvalConfig:
    project_name: str = "PatchCore-Inspection"
    model_version: str = "v1.0"
    dataset_version: str = "unknown"
    environment: str = ""
    author: str = "unknown"
    seed: int = 42
    output_dir: str = "outputs"
    fpr_max: float = 0.3
    num_thresholds: int = 1000
    run_command: str = "python scripts/eval.py configs/patchcore_mvtec.yaml"

    image_scores_path: str = ""
    image_labels_path: str = ""
    pixel_scores_path: str = ""
    pixel_masks_path: str = ""

    image_scores_key: Optional[str] = None
    image_labels_key: Optional[str] = None
    pixel_scores_key: Optional[str] = None
    pixel_masks_key: Optional[str] = None

    config_dict: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Dict[str, Any], output_override: Optional[str] = None) -> "EvalConfig":
        meta = data.get("meta", {})
        eval_cfg = data.get("eval", {})
        inputs = data.get("inputs", {})

        output_dir = output_override if output_override else eval_cfg.get("output_dir", "outputs")

        cfg = EvalConfig(
            project_name=meta.get("project_name", "PatchCore-Inspection"),
            model_version=meta.get("model_version", "v1.0"),
            dataset_version=meta.get("dataset_version", "unknown"),
            environment=meta.get("environment", ""),
            author=meta.get("author", "unknown"),
            seed=int(eval_cfg.get("seed", 42)),
            output_dir=str(output_dir),
            fpr_max=float(eval_cfg.get("fpr_max", 0.3)),
            num_thresholds=int(eval_cfg.get("num_thresholds", 1000)),
            run_command=eval_cfg.get(
                "run_command",
                "python scripts/eval.py configs/patchcore_mvtec.yaml",
            ),
            image_scores_path=str(inputs.get("image_scores", "")),
            image_labels_path=str(inputs.get("image_labels", "")),
            pixel_scores_path=str(inputs.get("pixel_scores", "")),
            pixel_masks_path=str(inputs.get("pixel_masks", "")),
            image_scores_key=inputs.get("image_scores_key"),
            image_labels_key=inputs.get("image_labels_key"),
            pixel_scores_key=inputs.get("pixel_scores_key"),
            pixel_masks_key=inputs.get("pixel_masks_key"),
            config_dict=data,
        )
        return cfg

    @staticmethod
    def from_patchcore_dict(
        data: Dict[str, Any], cfg_path: Optional[str] = None, output_override: Optional[str] = None
    ) -> "EvalConfig":
        report_cfg = data.get("report", {})
        eval_cfg = data.get("eval", {})
        model_cfg = data.get("model", {})
        data_cfg = data.get("data", {})

        output_dir = output_override
        if output_dir is None:
            output_dir = os.path.join(eval_cfg.get("save_dir", "outputs"), data_cfg.get("category", "default"))

        run_command = f"python scripts/eval.py {cfg_path}" if cfg_path else "python scripts/eval.py configs/patchcore_mvtec.yaml"

        return EvalConfig(
            project_name=report_cfg.get("project_name", "PatchCore-Inspection"),
            model_version=report_cfg.get("model_version", model_cfg.get("backbone", "unknown")),
            dataset_version=report_cfg.get("dataset_version", data_cfg.get("category", "unknown")),
            environment=report_cfg.get("environment", ""),
            author=report_cfg.get("author", "unknown"),
            seed=int(data.get("seed", 42)),
            output_dir=str(output_dir),
            fpr_max=float(eval_cfg.get("pro_fpr_max", 0.3)),
            num_thresholds=int(eval_cfg.get("num_thresholds", 1000)),
            run_command=report_cfg.get("run_command", run_command),
            config_dict=data,
        )


class EvalReportGenerator:
    def __init__(self, config: EvalConfig):
        self.cfg = config
        self.output_root = Path(config.output_dir)
        self.plots_dir = self.output_root / "plots"
        self.reports_dir = self.output_root / "reports"
        self.metrics_json_path = self.output_root / "metrics.json"

        self.report_id = self._make_report_id()
        self.run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.image_scores: Optional[np.ndarray] = None
        self.image_labels: Optional[np.ndarray] = None
        self.pixel_scores: Optional[np.ndarray] = None
        self.pixel_masks: Optional[np.ndarray] = None

        self.metrics: Dict[str, Any] = {}
        self.curves: Dict[str, Any] = {}

    def set_inputs(
        self,
        image_scores: np.ndarray,
        image_labels: np.ndarray,
        pixel_scores: np.ndarray,
        pixel_masks: np.ndarray,
    ) -> None:
        self.image_scores = np.asarray(image_scores)
        self.image_labels = np.asarray(image_labels)
        self.pixel_scores = np.asarray(pixel_scores)
        self.pixel_masks = np.asarray(pixel_masks)

    def load_results(self, metrics: Dict[str, Any], curves: Dict[str, Any]) -> None:
        self.metrics = metrics
        self.curves = curves

    def save_metrics_json(self) -> str:
        if not self.metrics:
            raise RuntimeError("Please compute or load metrics before saving metrics.json")
        with open(self.metrics_json_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": self.metrics, "curves": self.curves}, f, ensure_ascii=False, indent=2)
        return str(self.metrics_json_path)

    def _make_report_id(self) -> str:
        return f"PCR-{datetime.now().strftime('%Y%m%d')}-{random.getrandbits(24):06x}".upper()

    def _set_seed(self) -> None:
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def _ensure_dirs(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _load_array(self, path: str, key: Optional[str] = None) -> np.ndarray:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        suffix = p.suffix.lower()
        if suffix == ".npy":
            arr = np.load(p)
            return np.asarray(arr)
        if suffix == ".npz":
            data = np.load(p)
            if key:
                if key not in data:
                    raise KeyError(f"Key '{key}' not found in npz file: {path}")
                return np.asarray(data[key])
            first_key = list(data.keys())[0]
            return np.asarray(data[first_key])

        raise ValueError(f"Unsupported input format: {path}. Expected .npy or .npz")

    def _load_inputs(self) -> None:
        self.image_scores = self._load_array(self.cfg.image_scores_path, self.cfg.image_scores_key)
        self.image_labels = self._load_array(self.cfg.image_labels_path, self.cfg.image_labels_key)
        self.pixel_scores = self._load_array(self.cfg.pixel_scores_path, self.cfg.pixel_scores_key)
        self.pixel_masks = self._load_array(self.cfg.pixel_masks_path, self.cfg.pixel_masks_key)

    def _validate_inputs(self) -> None:
        if self.image_scores is None or self.image_labels is None:
            raise ValueError("image_scores and image_labels are required")
        if self.pixel_scores is None or self.pixel_masks is None:
            raise ValueError("pixel_scores and pixel_masks are required")

        if self.image_scores.ndim != 1:
            raise ValueError(f"image_scores must be 1D, got shape={self.image_scores.shape}")
        if self.image_labels.ndim != 1:
            raise ValueError(f"image_labels must be 1D, got shape={self.image_labels.shape}")
        if self.pixel_scores.ndim != 3:
            raise ValueError(f"pixel_scores must be 3D (N,H,W), got shape={self.pixel_scores.shape}")
        if self.pixel_masks.ndim != 3:
            raise ValueError(f"pixel_masks must be 3D (N,H,W), got shape={self.pixel_masks.shape}")

        n_img = self.image_scores.shape[0]
        if self.image_labels.shape[0] != n_img:
            raise ValueError(
                f"image_scores and image_labels length mismatch: {n_img} vs {self.image_labels.shape[0]}"
            )

        if self.pixel_scores.shape != self.pixel_masks.shape:
            raise ValueError(
                f"pixel_scores and pixel_masks shape mismatch: {self.pixel_scores.shape} vs {self.pixel_masks.shape}"
            )

        if self.pixel_scores.shape[0] != n_img:
            raise ValueError(
                "Image-level sample count and pixel-level sample count mismatch: "
                f"{n_img} vs {self.pixel_scores.shape[0]}"
            )

        unique_img_labels = np.unique(self.image_labels)
        if not np.all(np.isin(unique_img_labels, [0, 1])):
            raise ValueError("image_labels must contain only 0/1")

        unique_pixel_labels = np.unique(self.pixel_masks)
        if not np.all(np.isin(unique_pixel_labels, [0, 1])):
            raise ValueError("pixel_masks must contain only 0/1")

    def _compute_f1_curve(
        self, y_true: np.ndarray, y_score: np.ndarray, num_thresholds: int
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        score_min = float(np.min(y_score))
        score_max = float(np.max(y_score))
        if np.isclose(score_min, score_max):
            thresholds = np.array([score_min], dtype=float)
        else:
            thresholds = np.linspace(score_min, score_max, num_thresholds)

        f1_values = np.zeros_like(thresholds, dtype=float)
        y_true = y_true.astype(np.uint8)

        for i, t in enumerate(thresholds):
            y_pred = (y_score >= t).astype(np.uint8)
            tp = float(np.sum((y_pred == 1) & (y_true == 1)))
            fp = float(np.sum((y_pred == 1) & (y_true == 0)))
            fn = float(np.sum((y_pred == 0) & (y_true == 1)))
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1_values[i] = 2.0 * precision * recall / (precision + recall + 1e-12)

        best_idx = int(np.argmax(f1_values))
        best_f1 = float(f1_values[best_idx])
        best_thr = float(thresholds[best_idx])
        return thresholds, f1_values, best_f1, best_thr

    def _compute_pro(
        self,
        score_maps: np.ndarray,
        gt_masks: np.ndarray,
        num_thresholds: int,
        fpr_max: float,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute PRO under FPR constraint.

        Implementation details:
        1) Sweep thresholds from min(score_maps) to max(score_maps).
        2) For each threshold, binarize score maps.
        3) For each connected component in GT mask, compute overlap ratio:
           overlap = |pred ∩ region| / |region|
        4) PRO(threshold) = average overlap over all GT connected components.
        5) FPR(threshold) = false positive pixels on normal GT pixels / total normal pixels.
        6) Keep points where FPR <= fpr_max, sort by FPR, and compute normalized AUC:
           PRO_AUC = integral(PRO dFPR) / fpr_max
        """
        s_min = float(np.min(score_maps))
        s_max = float(np.max(score_maps))
        if np.isclose(s_min, s_max):
            thresholds = np.array([s_min], dtype=float)
        else:
            thresholds = np.linspace(s_min, s_max, num_thresholds)

        total_neg = float(np.sum(gt_masks == 0)) + 1e-12
        pros: List[float] = []
        fprs: List[float] = []

        for t in thresholds:
            pred_masks = (score_maps >= t).astype(np.uint8)

            fp_pixels = float(np.sum((pred_masks == 1) & (gt_masks == 0)))
            fpr = fp_pixels / total_neg

            region_overlaps: List[float] = []
            for pred, gt in zip(pred_masks, gt_masks):
                gt_labeled = label(gt.astype(np.uint8), connectivity=1)
                n_regions = int(gt_labeled.max())
                for rid in range(1, n_regions + 1):
                    region = (gt_labeled == rid)
                    region_area = float(np.sum(region))
                    if region_area <= 0:
                        continue
                    overlap = float(np.sum(pred[region] == 1)) / (region_area + 1e-12)
                    region_overlaps.append(overlap)

            if len(region_overlaps) == 0:
                continue

            pro_t = float(np.mean(region_overlaps))
            pros.append(pro_t)
            fprs.append(float(fpr))

        if len(pros) < 2:
            return 0.0, np.array([0.0]), np.array([0.0])

        fprs_arr = np.asarray(fprs, dtype=float)
        pros_arr = np.asarray(pros, dtype=float)
        order = np.argsort(fprs_arr)
        fprs_arr = fprs_arr[order]
        pros_arr = pros_arr[order]

        mask = fprs_arr <= fpr_max
        if int(np.sum(mask)) < 2:
            return 0.0, fprs_arr, pros_arr

        fprs_clip = fprs_arr[mask]
        pros_clip = pros_arr[mask]

        trapz_fn = getattr(np, "trapz", None)
        if trapz_fn is None:
            trapz_fn = np.trapezoid
        pro_auc = float(trapz_fn(pros_clip, fprs_clip) / (fpr_max + 1e-12))
        return pro_auc, fprs_arr, pros_arr

    def compute_metrics(self) -> Dict[str, Any]:
        if self.image_scores is None or self.image_labels is None:
            raise RuntimeError("Inputs not loaded")
        if self.pixel_scores is None or self.pixel_masks is None:
            raise RuntimeError("Inputs not loaded")

        image_scores = self.image_scores.astype(float)
        image_labels = self.image_labels.astype(np.uint8)

        pixel_scores_flat = self.pixel_scores.astype(float).reshape(-1)
        pixel_masks_flat = self.pixel_masks.astype(np.uint8).reshape(-1)

        img_fpr, img_tpr, _ = roc_curve(image_labels, image_scores)
        pix_fpr, pix_tpr, _ = roc_curve(pixel_masks_flat, pixel_scores_flat)

        img_auroc = float(roc_auc_score(image_labels, image_scores))
        pix_auroc = float(roc_auc_score(pixel_masks_flat, pixel_scores_flat))

        img_thresholds, img_f1_values, img_best_f1, img_best_thr = self._compute_f1_curve(
            image_labels, image_scores, self.cfg.num_thresholds
        )
        pix_thresholds, pix_f1_values, pix_best_f1, pix_best_thr = self._compute_f1_curve(
            pixel_masks_flat, pixel_scores_flat, self.cfg.num_thresholds
        )

        pro_auc, pro_fprs, pro_values = self._compute_pro(
            self.pixel_scores.astype(float),
            self.pixel_masks.astype(np.uint8),
            self.cfg.num_thresholds,
            self.cfg.fpr_max,
        )

        img_pred = (image_scores >= img_best_thr).astype(np.uint8)
        cm = confusion_matrix(image_labels, img_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]

        precision = float(tp / (tp + fp + 1e-12))
        recall = float(tp / (tp + fn + 1e-12))
        specificity = float(tn / (tn + fp + 1e-12))
        accuracy = float((tp + tn) / (tp + tn + fp + fn + 1e-12))

        self.curves = {
            "img_roc": {"fpr": img_fpr.tolist(), "tpr": img_tpr.tolist()},
            "pix_roc": {"fpr": pix_fpr.tolist(), "tpr": pix_tpr.tolist()},
            "img_f1": {"thresholds": img_thresholds.tolist(), "f1": img_f1_values.tolist()},
            "pix_f1": {"thresholds": pix_thresholds.tolist(), "f1": pix_f1_values.tolist()},
            "pro": {"fpr": pro_fprs.tolist(), "pro": pro_values.tolist()},
        }

        self.metrics = {
            "report_id": self.report_id,
            "date": self.run_date,
            "image_auroc": img_auroc,
            "pixel_auroc": pix_auroc,
            "image_f1_max": img_best_f1,
            "image_f1_threshold": img_best_thr,
            "pixel_f1_max": pix_best_f1,
            "pixel_f1_threshold": pix_best_thr,
            "pro_auc_fpr_le_0_3": pro_auc,
            "pro_fpr_max": self.cfg.fpr_max,
            "confusion_matrix": {
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            },
            "derived": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
            },
        }
        return self.metrics

    def generate_plots(self) -> Dict[str, str]:
        """Generate ROC, F1, and confusion matrix plots using metrics.py functions."""
        if not self.metrics or not self.curves:
            raise RuntimeError("Please run compute_metrics() before generate_plots()")

        self._ensure_dirs()

        # Prepare output paths
        roc_path = str(self.plots_dir / "roc_curve.png")
        f1_path = str(self.plots_dir / "f1_curve.png")
        cm_path = str(self.plots_dir / "confusion_matrix.png")

        # Extract ROC curve data
        img_fpr = np.asarray(self.curves["img_roc"]["fpr"], dtype=float)
        img_tpr = np.asarray(self.curves["img_roc"]["tpr"], dtype=float)
        pix_fpr = np.asarray(self.curves["pix_roc"]["fpr"], dtype=float)
        pix_tpr = np.asarray(self.curves["pix_roc"]["tpr"], dtype=float)

        # Call plot_roc_curve from metrics.py
        plot_roc_curve(
            curves=[
                {
                    "name": "Image-level",
                    "fprs": img_fpr,
                    "tprs": img_tpr,
                    "auroc": self.metrics["image_auroc"],
                    "color": "#0072b2",
                },
                {
                    "name": "Pixel-level",
                    "fprs": pix_fpr,
                    "tprs": pix_tpr,
                    "auroc": self.metrics["pixel_auroc"],
                    "color": "#d55e00",
                },
            ],
            save_path=roc_path,
            title="ROC Curve",
        )

        # Extract F1 curve data
        img_thresholds = np.asarray(self.curves["img_f1"]["thresholds"], dtype=float)
        img_f1_values = np.asarray(self.curves["img_f1"]["f1"], dtype=float)
        pix_thresholds = np.asarray(self.curves["pix_f1"]["thresholds"], dtype=float)
        pix_f1_values = np.asarray(self.curves["pix_f1"]["f1"], dtype=float)

        # Call plot_f1_curve from metrics.py
        plot_f1_curve(
            curves=[
                {
                    "name": "Image-level",
                    "thresholds": img_thresholds,
                    "f1_values": img_f1_values,
                    "best_f1": self.metrics["image_f1_max"],
                    "best_t": self.metrics["image_f1_threshold"],
                    "color": "#0072b2",
                },
                {
                    "name": "Pixel-level",
                    "thresholds": pix_thresholds,
                    "f1_values": pix_f1_values,
                    "best_f1": self.metrics["pixel_f1_max"],
                    "best_t": self.metrics["pixel_f1_threshold"],
                    "color": "#d55e00",
                },
            ],
            save_path=f1_path,
            title="F1-Confidence Curve",
        )

        # Prepare confusion matrix
        cm_dict = self.metrics["confusion_matrix"]
        cm = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]], dtype=int)

        # Prepare metrics for display
        metric_display = {
            "Accuracy": self.metrics["derived"]["accuracy"],
            "Precision": self.metrics["derived"]["precision"],
            "Recall": self.metrics["derived"]["recall"],
            "Specificity": self.metrics["derived"]["specificity"],
            "Image AUROC": self.metrics["image_auroc"],
            "Pixel AUROC": self.metrics["pixel_auroc"],
            "PRO": self.metrics["pro_auc_fpr_le_0_3"],
        }

        # Call plot_confusion_matrix from metrics.py
        plot_confusion_matrix(
            cm=cm,
            save_path=cm_path,
            title="Confusion Matrix",
            metrics=metric_display,
        )

        return {
            "roc_curve": roc_path,
            "f1_curve": f1_path,
            "confusion_matrix": cm_path,
        }

    def _build_conclusion_cn(self) -> str:
        m = self.metrics
        notes: List[str] = []
        if m["image_auroc"] >= 0.95:
            notes.append("图像级判别能力优秀。")
        elif m["image_auroc"] >= 0.85:
            notes.append("图像级判别能力良好。")
        else:
            notes.append("图像级判别能力偏弱，建议优化特征提取或阈值策略。")

        if m["pixel_f1_max"] < 0.70:
            notes.append("像素级定位能力偏弱（Pixel F1 < 0.70），建议增强分割后处理或引入多尺度特征。")
        else:
            notes.append("像素级定位能力达到可用水平。")

        if m["pro_auc_fpr_le_0_3"] < 0.60:
            notes.append("在低误报约束下（FPR<=0.3）的区域覆盖表现一般，可关注误检抑制。")
        else:
            notes.append("低误报约束下区域覆盖表现良好。")

        return " ".join(notes)

    def _build_conclusion_en(self) -> str:
        m = self.metrics
        notes: List[str] = []
        if m["image_auroc"] >= 0.95:
            notes.append("Image-level discrimination is excellent.")
        elif m["image_auroc"] >= 0.85:
            notes.append("Image-level discrimination is good.")
        else:
            notes.append("Image-level discrimination is relatively weak; consider improving feature extraction or thresholding.")

        if m["pixel_f1_max"] < 0.70:
            notes.append("Pixel-level localization is weak (Pixel F1 < 0.70); consider stronger post-processing or multi-scale features.")
        else:
            notes.append("Pixel-level localization is at a usable level.")

        if m["pro_auc_fpr_le_0_3"] < 0.60:
            notes.append("Region overlap under low-FPR constraint (FPR<=0.3) is moderate; false-positive suppression can be improved.")
        else:
            notes.append("Region overlap under low-FPR constraint is strong.")

        return " ".join(notes)

    def _format_config_text(self) -> str:
        return json.dumps(self.cfg.config_dict, ensure_ascii=False, indent=2)

    def _pdf_common(
        self,
        output_pdf_path: Path,
        language: str,
        plots: Dict[str, str],
    ) -> None:
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import cm
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.cidfonts import UnicodeCIDFont
            from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle
        except Exception as e:
            raise ImportError("reportlab is required for PDF generation. Please install reportlab.") from e

        if language == "cn":
            try:
                pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
                base_font = "STSong-Light"
            except Exception:
                base_font = "Helvetica"
        else:
            base_font = "Helvetica"

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontName=base_font, fontSize=18, leading=22)
        heading_style = ParagraphStyle("HeadingStyle", parent=styles["Heading2"], fontName=base_font, fontSize=13, leading=16)
        body_style = ParagraphStyle("BodyStyle", parent=styles["BodyText"], fontName=base_font, fontSize=10, leading=14)
        code_style = ParagraphStyle("CodeStyle", parent=styles["Code"], fontName=base_font, fontSize=8.5, leading=10)

        doc = SimpleDocTemplate(str(output_pdf_path), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
        elems: List[Any] = []

        if language == "cn":
            title = "PatchCore 评估报告"
            summary_header = "Summary（关键指标）"
            config_header = "Configuration（配置）"
            metrics_def_header = "Metrics Definition（指标定义）"
            viz_header = "Visualization（可视化）"
            conclusion_header = "Conclusion（结论）"
            appendix_header = "Appendix（附录）"
            conclusion = self._build_conclusion_cn()
            metrics_defs = [
                "Image AUROC：图像级异常分数与标签的ROC面积。",
                "Pixel AUROC：像素级分数图与mask展开后的ROC面积。",
                "Image F1(max-f1)：图像级阈值扫描后可达到的最大F1。",
                "Pixel F1(max-f1)：像素级阈值扫描后可达到的最大F1。",
                f"PRO（FPR<={self.cfg.fpr_max}）：对连通区域重叠率曲线在低FPR区间做归一化面积。",
            ]
            author_label = "作者"
            date_label = "日期"
            env_label = "运行环境"
            report_id_label = "Report ID"
            model_label = "模型版本"
            dataset_label = "数据集版本"
        else:
            title = "PatchCore Evaluation Report"
            summary_header = "Summary"
            config_header = "Configuration"
            metrics_def_header = "Metrics Definition"
            viz_header = "Visualization"
            conclusion_header = "Conclusion"
            appendix_header = "Appendix"
            conclusion = self._build_conclusion_en()
            metrics_defs = [
                "Image AUROC: ROC-AUC computed from image-level anomaly scores and labels.",
                "Pixel AUROC: ROC-AUC computed from flattened pixel scores and GT masks.",
                "Image F1 (max-f1): maximum F1 by threshold scanning on image-level scores.",
                "Pixel F1 (max-f1): maximum F1 by threshold scanning on pixel-level scores.",
                f"PRO (FPR<={self.cfg.fpr_max}): normalized area under PRO-FPR curve in the low-FPR range.",
            ]
            author_label = "Author"
            date_label = "Date"
            env_label = "Environment"
            report_id_label = "Report ID"
            model_label = "Model Version"
            dataset_label = "Dataset Version"

        elems.append(Paragraph(title, title_style))
        elems.append(Spacer(1, 8))

        env_text = self.cfg.environment or f"Python {sys.version.split()[0]} | {platform.platform()}"
        meta_rows = [
            [report_id_label, self.report_id],
            [date_label, self.run_date],
            ["Project", self.cfg.project_name],
            [model_label, self.cfg.model_version],
            [dataset_label, self.cfg.dataset_version],
            [env_label, env_text],
            [author_label, self.cfg.author],
        ]
        meta_table = Table(meta_rows, colWidths=[4.0 * cm, 11.5 * cm])
        meta_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), base_font),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        elems.append(meta_table)
        elems.append(Spacer(1, 10))

        elems.append(Paragraph(summary_header, heading_style))
        summary_rows = [
            ["Metric", "Value"],
            ["Image AUROC", f"{self.metrics['image_auroc']:.6f}"],
            ["Pixel AUROC", f"{self.metrics['pixel_auroc']:.6f}"],
            ["Image F1(max-f1)", f"{self.metrics['image_f1_max']:.6f}"],
            ["Image F1 threshold", f"{self.metrics['image_f1_threshold']:.6f}"],
            ["Pixel F1(max-f1)", f"{self.metrics['pixel_f1_max']:.6f}"],
            ["Pixel F1 threshold", f"{self.metrics['pixel_f1_threshold']:.6f}"],
            [f"PRO (FPR<={self.cfg.fpr_max})", f"{self.metrics['pro_auc_fpr_le_0_3']:.6f}"],
            ["Accuracy", f"{self.metrics['derived']['accuracy']:.6f}"],
            ["Precision", f"{self.metrics['derived']['precision']:.6f}"],
            ["Recall", f"{self.metrics['derived']['recall']:.6f}"],
            ["Specificity", f"{self.metrics['derived']['specificity']:.6f}"],
        ]
        summary_table = Table(summary_rows, colWidths=[7.0 * cm, 8.5 * cm])
        summary_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), base_font),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ]
            )
        )
        elems.append(summary_table)
        elems.append(Spacer(1, 10))

        elems.append(Paragraph(config_header, heading_style))
        config_text = self._format_config_text()
        elems.append(Preformatted(config_text, code_style))
        elems.append(Spacer(1, 8))

        elems.append(Paragraph(metrics_def_header, heading_style))
        for d in metrics_defs:
            elems.append(Paragraph(f"- {d}", body_style))
        elems.append(Spacer(1, 8))

        elems.append(PageBreak())
        elems.append(Paragraph(viz_header, heading_style))
        elems.append(Spacer(1, 6))

        for name in ["roc_curve", "f1_curve", "confusion_matrix"]:
            img_path = plots[name]
            elems.append(Paragraph(name, body_style))
            elems.append(Spacer(1, 4))
            elems.append(Image(img_path, width=16.0 * cm, height=10.5 * cm))
            elems.append(Spacer(1, 10))

        elems.append(Paragraph(conclusion_header, heading_style))
        elems.append(Paragraph(conclusion, body_style))
        elems.append(Spacer(1, 8))

        elems.append(Paragraph(appendix_header, heading_style))
        file_list = [
            str(self.metrics_json_path),
            plots["roc_curve"],
            plots["f1_curve"],
            plots["confusion_matrix"],
            str(self.reports_dir / "eval_report_cn.pdf"),
            str(self.reports_dir / "eval_report_en.pdf"),
        ]
        if language == "cn":
            elems.append(Paragraph("输出文件清单：", body_style))
            elems.append(Paragraph("复现命令：", body_style))
        else:
            elems.append(Paragraph("Output file list:", body_style))
            elems.append(Paragraph("Reproducible command:", body_style))

        for fp in file_list:
            elems.append(Paragraph(f"- {fp}", body_style))

        elems.append(Paragraph(self.cfg.run_command, code_style))

        doc.build(elems)

    def generate_pdf_cn(self, plots: Dict[str, str]) -> str:
        pdf_path = self.reports_dir / "eval_report_cn.pdf"
        self._pdf_common(pdf_path, language="cn", plots=plots)
        return str(pdf_path)

    def generate_pdf_en(self, plots: Dict[str, str]) -> str:
        pdf_path = self.reports_dir / "eval_report_en.pdf"
        self._pdf_common(pdf_path, language="en", plots=plots)
        return str(pdf_path)

    def run_all(self) -> Dict[str, Any]:
        self._set_seed()
        self._ensure_dirs()
        self._load_inputs()
        self._validate_inputs()

        print("[EvalReport] Computing metrics...")
        metrics = self.compute_metrics()

        print("[EvalReport] Saving metrics.json...")
        self.save_metrics_json()

        print("[EvalReport] Generating plots...")
        plots = self.generate_plots()

        print("[EvalReport] Generating Chinese PDF...")
        pdf_cn = self.generate_pdf_cn(plots)

        print("[EvalReport] Generating English PDF...")
        pdf_en = self.generate_pdf_en(plots)

        print("[EvalReport] Done")
        return {
            "metrics_json": str(self.metrics_json_path),
            "roc_curve": plots["roc_curve"],
            "f1_curve": plots["f1_curve"],
            "confusion_matrix": plots["confusion_matrix"],
            "report_cn": pdf_cn,
            "report_en": pdf_en,
        }


def load_config_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    if p.suffix.lower() in [".yaml", ".yml"]:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError("Config file must be yaml/yml/json")


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchCore evaluation report generator")
    parser.add_argument("--config", type=str, required=True, help="Path to PatchCore config file (yaml/json)")
    parser.add_argument("--output", type=str, default=None, help="Output root directory")
    parser.add_argument("--image-scores", type=str, default=None, help="Path to image_scores.npy or .npz")
    parser.add_argument("--image-labels", type=str, default=None, help="Path to image_labels.npy or .npz")
    parser.add_argument("--pixel-scores", type=str, default=None, help="Path to pixel_scores.npy or .npz")
    parser.add_argument("--pixel-masks", type=str, default=None, help="Path to pixel_masks.npy or .npz")
    parser.add_argument("--image-scores-key", type=str, default=None, help="Optional key for npz image scores")
    parser.add_argument("--image-labels-key", type=str, default=None, help="Optional key for npz image labels")
    parser.add_argument("--pixel-scores-key", type=str, default=None, help="Optional key for npz pixel scores")
    parser.add_argument("--pixel-masks-key", type=str, default=None, help="Optional key for npz pixel masks")
    args = parser.parse_args()

    data = load_config_file(args.config)
    cfg = EvalConfig.from_patchcore_dict(data, cfg_path=args.config, output_override=args.output)
    if args.image_scores:
        cfg.image_scores_path = args.image_scores
    if args.image_labels:
        cfg.image_labels_path = args.image_labels
    if args.pixel_scores:
        cfg.pixel_scores_path = args.pixel_scores
    if args.pixel_masks:
        cfg.pixel_masks_path = args.pixel_masks
    cfg.image_scores_key = args.image_scores_key
    cfg.image_labels_key = args.image_labels_key
    cfg.pixel_scores_key = args.pixel_scores_key
    cfg.pixel_masks_key = args.pixel_masks_key

    if not all([cfg.image_scores_path, cfg.image_labels_path, cfg.pixel_scores_path, cfg.pixel_masks_path]):
        raise ValueError(
            "When running eval_report_generator.py standalone, you must provide --image-scores, --image-labels, --pixel-scores, and --pixel-masks. "
            "For the integrated workflow, run python scripts/eval.py <patchcore_config>."
        )

    generator = EvalReportGenerator(cfg)
    outputs = generator.run_all()

    print("[EvalReport] Output files:")
    for k, v in outputs.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
