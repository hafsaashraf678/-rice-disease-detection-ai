"""
scripts/95_sheath_blight_improved_estimator.py
================================================
PHASE 9: Sheath Blight — Upgraded High-Precision Estimator Class
Audited & Upgraded for Full Sheath & Lesion Boundary Retention
"""

import cv2
import numpy as np


class SheathBlightImprovedEstimator:

    def __init__(self):
        # HSV range for Sheath Blight lesions: water-soaked green-gray centers with brown borders
        self.hsv_lower = np.array([0,   12,  20], dtype=np.uint8)
        self.hsv_upper = np.array([42, 255, 230], dtype=np.uint8)
        self.min_blob_px = 15
        self.min_leaf_frac = 0.015

    def estimate_severity(self, image_bgr: np.ndarray, leaf_mask: np.ndarray = None) -> dict:
        h, w = image_bgr.shape[:2]
        tot_px = h * w

        if leaf_mask is None:
            leaf_mask = self._extract_leaf_roi(image_bgr)

        leaf_bin = (leaf_mask > 0).astype(np.uint8)
        leaf_px  = int(np.sum(leaf_bin))

        if leaf_px == 0:
            return self._empty_result(image_bgr.shape[:2])

        raw_mask           = self._generate_raw_mask(image_bgr, leaf_bin)
        final_mask, rem_px = self._filter_components(raw_mask, leaf_bin, leaf_px)

        lesion_px    = int(np.sum(final_mask > 0))
        severity_pct = round(lesion_px / float(leaf_px) * 100.0, 2)
        grade, label = self._get_ses_grade(severity_pct)

        return {
            "leaf_mask"         : leaf_mask,
            "raw_hsv_mask"      : raw_mask,
            "final_lesion_mask" : final_mask,
            "leaf_pixels"       : leaf_px,
            "lesion_pixels"     : lesion_px,
            "removed_pixels"    : rem_px,
            "severity_pct"      : severity_pct,
            "ses_grade"         : grade,
            "severity_label"    : label,
        }

    def _extract_leaf_roi(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        tot_px = float(h * w)

        hsv  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Full sheath spectrum (Green healthy + Straw-colored infected sheath tissue)
        green_mask  = cv2.inRange(hsv, np.array([22, 10, 15]),  np.array([92, 255, 255]))
        sheath_mask = cv2.inRange(hsv, np.array([2,  10, 20]),  np.array([48, 255, 255]))
        plant_color = cv2.bitwise_or(green_mask, sheath_mask)

        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean  = cv2.morphologyEx(plant_color, cv2.MORPH_OPEN, k_open)

        cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled  = np.zeros((h, w), dtype=np.uint8)

        if cnts:
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area >= 150:
                    cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)

        if np.sum(filled > 0) / tot_px < self.min_leaf_frac:
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            sat_mask = (hsv[:, :, 1] > 12).astype(np.uint8) * 255
            gated_otsu = cv2.bitwise_and(otsu, plant_color)
            gated_otsu = cv2.bitwise_and(gated_otsu, sat_mask)
            k_m = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            filled = cv2.morphologyEx(gated_otsu, cv2.MORPH_OPEN, k_m)

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        return cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k_close)

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

            if area < self.min_blob_px and asp < 1.1:
                rem_px += area
                continue

            out[blob > 0] = 255

        return out, rem_px

    def _get_ses_grade(self, pct: float):
        if pct == 0.0:
            return 0, "Grade 0 (0% — Healthy)"
        elif pct <= 1.0:
            return 1, "Grade 1 (0.01-1% — Highly Resistant)"
        elif pct <= 5.0:
            return 3, "Grade 3 (1.01-5% — Moderately Resistant)"
        elif pct <= 25.0:
            return 5, "Grade 5 (5.01-25% — Moderately Susceptible)"
        elif pct <= 50.0:
            return 7, "Grade 7 (25.01-50% — Susceptible)"
        else:
            return 9, "Grade 9 (>50% — Very Severe / Whole Sheath Rotted)"

    def _empty_result(self, shape):
        z = np.zeros(shape, dtype=np.uint8)
        return {
            "leaf_mask": z, "raw_hsv_mask": z, "final_lesion_mask": z,
            "leaf_pixels": 0, "lesion_pixels": 0, "removed_pixels": 0,
            "severity_pct": 0.0, "ses_grade": 0, "severity_label": "Grade 0 (0% — Healthy)"
        }
