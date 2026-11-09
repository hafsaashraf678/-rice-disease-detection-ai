"""
scripts/46_rice_blast_severity_estimator.py
=============================================
PHASE 4: Rice Blast — Standalone Severity Estimator Class
Audited & Verified Method (RiceBlastFixEstimator)

Algorithm Pipeline:
  1. Primary ROI Extraction: ExG (Excess Green: 2G - R - B) combined with
     HSV Color Saliency (Green: H in [25, 85], Yellow/Tan: H in [10, 25]) +
     Morphological closing + Oversized Mask Rejection (>70% frame area).
  2. Fallback ROI: Central-Focused Color Fallback (10% padding + plant color mask).
  3. HSV Lesion Extraction: H in [0, 30], S in [30, 255], V in [20, 180] restricted
     strictly inside the valid leaf ROI.
  4. Connected Components Filtering: Removes noise specks (< 12 px) and oversized blobs (> 35% of leaf).
  5. Severity Calculation: (lesion_pixels / valid_leaf_pixels) * 100.
  6. IRRI Standard Evaluation System (SES Scale 0-9):
       Grade 0 : 0.00%           → Healthy / No Lesion
       Grade 1 : 0.01% - 1.00%   → Small pin-point specks
       Grade 3 : 1.01% - 5.00%   → Small round-to-elongate necrotic spots
       Grade 5 : 5.01% - 25.00%  → Typical spindle-shaped blast lesions (Moderate)
       Grade 7 : 25.01% - 50.00% → Severe lesions (25-50% leaf area)
       Grade 9 : > 50.00%        → Extreme / whole leaf killed
"""

import cv2
import numpy as np


class RiceBlastSeverityEstimator:

    def __init__(self):
        self.lesion_hsv_lower    = np.array([0,   30,  20], dtype=np.uint8)
        self.lesion_hsv_upper    = np.array([30, 255, 180], dtype=np.uint8)
        self.min_lesion_px       = 12
        self.max_frame_roi_ratio = 0.70  # Reject ROI > 70% frame

    def extract_leaf_roi(self, image_bgr: np.ndarray, precomputed_mask: np.ndarray = None) -> tuple:
        h, w = image_bgr.shape[:2]
        total_px = h * w

        if precomputed_mask is not None and isinstance(precomputed_mask, np.ndarray):
            roi_bin = (precomputed_mask > 0).astype(np.uint8)
            roi_px  = int(np.sum(roi_bin))
            if 0.05 <= (roi_px / float(total_px)) <= self.max_frame_roi_ratio:
                return cv2.resize(roi_bin * 255, (w, h)), "SAM2 / Precomputed ROI"

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        b, g, r = cv2.split(image_bgr.astype(np.float32))
        exg     = 2.0 * g - r - b
        exg_n   = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, exg_bin = cv2.threshold(exg_n, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        green_mask  = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([85, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([10, 25, 30]), np.array([25, 255, 240]))
        plant_color = cv2.bitwise_or(green_mask, yellow_mask)

        raw_plant = cv2.bitwise_and(exg_bin, plant_color)

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean  = cv2.morphologyEx(raw_plant, cv2.MORPH_OPEN, k_open)

        cnts, _    = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_roi = np.zeros((h, w), dtype=np.uint8)

        if cnts:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < max(400, total_px * 0.004):
                    continue
                if (area / float(total_px)) > self.max_frame_roi_ratio:
                    continue
                cv2.drawContours(target_roi, [cnt], -1, 255, thickness=cv2.FILLED)

        k_close   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        final_roi = cv2.morphologyEx(target_roi, cv2.MORPH_CLOSE, k_close)

        roi_px    = int(np.sum(final_roi > 0))
        roi_ratio = roi_px / float(total_px)

        if roi_px == 0 or roi_ratio > self.max_frame_roi_ratio:
            center_mask = np.zeros((h, w), dtype=np.uint8)
            pad_h, pad_w = int(h * 0.10), int(w * 0.10)
            center_mask[pad_h:h-pad_h, pad_w:w-pad_w] = 255
            final_roi = cv2.bitwise_and(plant_color, center_mask)
            method = "Central-Focused Color Fallback"
        else:
            method = "Hybrid ExG+Color Geometry ROI"

        return final_roi, method

    def estimate_severity(self, image_bgr: np.ndarray, precomputed_mask: np.ndarray = None) -> dict:
        h, w = image_bgr.shape[:2]
        leaf_mask, method = self.extract_leaf_roi(image_bgr, precomputed_mask)
        leaf_bin = (leaf_mask > 0).astype(np.uint8)
        leaf_px  = int(np.sum(leaf_bin))

        if leaf_px == 0:
            return self._empty_result((h, w), method)

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        raw_lesion = cv2.inRange(hsv, self.lesion_hsv_lower, self.lesion_hsv_upper)

        masked_lesion = cv2.bitwise_and(raw_lesion, raw_lesion, mask=leaf_bin)

        k_sm      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_raw = cv2.morphologyEx(masked_lesion, cv2.MORPH_OPEN, k_sm)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(clean_raw, connectivity=8)
        final_lesion = np.zeros_like(clean_raw)
        n_components = 0

        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_lesion_px:
                continue

            if (area / float(leaf_px)) > 0.35:
                continue

            blob = (labels == i).astype(np.uint8)
            final_lesion[blob > 0] = 255
            n_components += 1

        final_lesion_px = int(np.sum(final_lesion > 0))

        roi_pct          = round(leaf_px / float(h * w) * 100.0, 2)
        final_lesion_pct = round(final_lesion_px / float(leaf_px) * 100.0, 2) if leaf_px > 0 else 0.0

        ses_grade, label = self._get_ses_grade(final_lesion_pct)

        return {
            "leaf_mask"         : leaf_mask,
            "raw_hsv_mask"      : clean_raw,
            "final_lesion_mask" : final_lesion,
            "leaf_pixels"       : leaf_px,
            "lesion_pixels"     : final_lesion_px,
            "severity_pct"      : final_lesion_pct,
            "roi_pct"           : roi_pct,
            "ses_grade"         : ses_grade,
            "severity_label"    : label,
            "n_components"      : n_components,
            "method_used"       : method,
        }

    def _get_ses_grade(self, pct: float) -> tuple:
        if pct == 0.0:
            return 0, "Grade 0 (0% — Healthy)"
        elif pct <= 1.0:
            return 1, "Grade 1 (0.01-1% — Pin-point Specks)"
        elif pct <= 5.0:
            return 3, "Grade 3 (1-5% — Small Lesions)"
        elif pct <= 25.0:
            return 5, "Grade 5 (5-25% — Spindle Lesions / Moderate)"
        elif pct <= 50.0:
            return 7, "Grade 7 (25-50% — Severe Lesions)"
        else:
            return 9, "Grade 9 (>50% — Extreme / Whole Leaf Killed)"

    def _empty_result(self, shape, method):
        z = np.zeros(shape, dtype=np.uint8)
        return {
            "leaf_mask": z, "raw_hsv_mask": z, "final_lesion_mask": z,
            "leaf_pixels": 0, "lesion_pixels": 0, "severity_pct": 0.0,
            "roi_pct": 0.0, "ses_grade": 0, "severity_label": "Grade 0 (0% — Healthy)",
            "n_components": 0, "method_used": method
        }
