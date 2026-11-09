"""
scripts/41_nbls_severity_estimator.py
======================================
PHASE 4: Narrow Brown Leaf Spot (NBLS) — Standalone Severity Estimator Class

PERCENT DISEASE SEVERITY (PDS%) / IRRI RICE SEVERITY SCALE:
  Grade 0 : 0.00%           → Healthy / No Lesion
  Grade 1 : 0.01% - 1.00%   → Very Low / Mild / Resistant
  Grade 3 : 1.01% - 5.00%   → Low-Moderate / Moderately Susceptible
  Grade 5 : 5.01% - 25.00%  → Moderate-Severe / Susceptible
  Grade 7 : 25.01% - 50.00% → Very Severe / Highly Susceptible
  Grade 9 : > 50.00%        → Extreme / Completely Susceptible
"""

import cv2
import numpy as np


class NBLSSeverityEstimator:

    def __init__(self):
        # HSV range for NBLS lesions (Variant A)
        self.hsv_lower = np.array([0,  25, 20],  dtype=np.uint8)
        self.hsv_upper = np.array([35, 255, 170], dtype=np.uint8)
        self.min_blob_px = 15
        self.max_leaf_frac = 0.50

    def estimate_severity(self, image_bgr: np.ndarray, leaf_mask: np.ndarray = None) -> dict:
        if leaf_mask is None:
            leaf_mask = self._generate_fallback_leaf_mask(image_bgr)

        leaf_bin = (leaf_mask > 0).astype(np.uint8)
        leaf_px  = int(np.sum(leaf_bin))

        if leaf_px == 0:
            return self._empty_result(image_bgr.shape[:2])

        raw_mask           = self._generate_raw_mask(image_bgr, leaf_bin)
        final_mask, rem_px = self._filter_components(raw_mask, leaf_bin, leaf_px)

        lesion_px    = int(np.sum(final_mask > 0))
        severity_pct = round(lesion_px / float(leaf_px) * 100.0, 2)
        grade, label = self._get_pds_grade(severity_pct)

        return {
            "leaf_mask"         : leaf_mask,
            "raw_hsv_mask"      : raw_mask,
            "final_lesion_mask" : final_mask,
            "leaf_pixels"       : leaf_px,
            "lesion_pixels"     : lesion_px,
            "removed_pixels"    : rem_px,
            "severity_pct"      : severity_pct,
            "pds_grade"         : grade,
            "severity_label"    : label,
        }

    def _generate_fallback_leaf_mask(self, image_bgr):
        hsv  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        green_mask  = cv2.inRange(hsv, np.array([25, 20, 20]),  np.array([85, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([10, 15, 30]),  np.array([40, 255, 255]))
        combined    = cv2.bitwise_or(green_mask, yellow_mask)

        _, otsu     = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        raw_leaf    = cv2.bitwise_or(combined, otsu)

        kernel_sm   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean       = cv2.morphologyEx(raw_leaf, cv2.MORPH_OPEN, kernel_sm)

        cnts, _     = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled      = np.zeros_like(clean)
        if cnts:
            for cnt in cnts:
                if cv2.contourArea(cnt) >= 500:
                    cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)

        kernel_lg   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        return cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_lg)

    def _generate_raw_mask(self, image_bgr, leaf_bin):
        hsv    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        raw    = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        masked = cv2.bitwise_and(raw, raw, mask=leaf_bin)
        k      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(masked, cv2.MORPH_OPEN, k)

    def _filter_components(self, raw_mask, leaf_bin, leaf_px):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
        out    = np.zeros_like(raw_mask)
        rem_px = 0

        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            blob = (labels == i).astype(np.uint8)

            w_b, h_b = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            asp      = max(w_b, h_b) / (min(w_b, h_b) + 1e-5)

            if area < self.min_blob_px and asp < 1.5:
                rem_px += area
                continue

            if leaf_px > 0 and (area / float(leaf_px)) > self.max_leaf_frac:
                rem_px += area
                continue

            out[blob > 0] = 255

        return out, rem_px

    def _get_pds_grade(self, pct: float):
        if pct == 0.0:
            return 0, "Grade 0 (0% — Healthy)"
        elif pct <= 1.0:
            return 1, "Grade 1 (0.01-1% — Very Low)"
        elif pct <= 5.0:
            return 3, "Grade 3 (1-5% — Mild)"
        elif pct <= 25.0:
            return 5, "Grade 5 (5-25% — Moderate)"
        elif pct <= 50.0:
            return 7, "Grade 7 (25-50% — Severe)"
        else:
            return 9, "Grade 9 (>50% — Extreme)"

    def _empty_result(self, shape):
        z = np.zeros(shape, dtype=np.uint8)
        return {
            "leaf_mask": z, "raw_hsv_mask": z, "final_lesion_mask": z,
            "leaf_pixels": 0, "lesion_pixels": 0, "removed_pixels": 0,
            "severity_pct": 0.0, "pds_grade": 0, "severity_label": "Grade 0 (0% — Healthy)"
        }
