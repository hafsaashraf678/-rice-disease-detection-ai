"""
scripts/25_bacterial_blight_severity_estimator.py
===================================================
Standalone Bacterial Leaf Blight (BLB) Severity Estimator Class.

RESEARCH PAPER IRRI RICE DISEASE RATING SCALE:
  • Rating 0 : 0.00%           (No Lesion / Healthy)
  • Rating 1 : 1% - 10%        (Low / Mild)
  • Rating 3 : 11% - 30%       (Moderate)
  • Rating 5 : 31% - 50%       (Severe)
  • Rating 7 : 51% - 75%       (Very Severe)
  • Rating 9 : 76% - 100%      (Extreme / Critical)
"""

import cv2
import numpy as np


class BacterialBlightSeverityEstimator:
    def __init__(self):
        self.hsv_lower = np.array([10, 20, 35], dtype=np.uint8)
        self.hsv_upper = np.array([38, 255, 240], dtype=np.uint8)
        self.min_blob_px = 25

    def estimate_severity(self, image_bgr: np.ndarray, leaf_mask: np.ndarray = None) -> dict:
        if leaf_mask is None:
            leaf_mask = self._generate_fallback_leaf_mask(image_bgr)

        leaf_bin = (leaf_mask > 0).astype(np.uint8)
        leaf_px  = int(np.sum(leaf_bin))

        if leaf_px == 0:
            return self._empty_result(image_bgr.shape[:2])

        varA_mask  = self._generate_variant_a_mask(image_bgr, leaf_bin)
        final_mask, rem_px = self._filter_blb_components(varA_mask, leaf_bin, leaf_px)

        lesion_px    = int(np.sum(final_mask > 0))
        severity_pct = round(lesion_px / float(leaf_px) * 100.0, 2)
        disease_rating, rating_label = self._get_paper_disease_rating(severity_pct)

        return {
            "leaf_mask"          : leaf_mask,
            "raw_hsv_mask"       : varA_mask,
            "final_lesion_mask"  : final_mask,
            "leaf_pixels"        : leaf_px,
            "lesion_pixels"      : lesion_px,
            "removed_pixels"     : rem_px,
            "severity_pct"       : severity_pct,
            "disease_rating"     : disease_rating,
            "rating_label"       : rating_label,
        }

    def _generate_fallback_leaf_mask(self, image_bgr):
        hsv  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        green_mask   = cv2.inRange(hsv, np.array([25, 20, 20]),  np.array([85, 255, 255]))
        yellow_mask  = cv2.inRange(hsv, np.array([10, 15, 30]),  np.array([40, 255, 255]))
        combined_col = cv2.bitwise_or(green_mask, yellow_mask)

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        raw_leaf = cv2.bitwise_or(combined_col, otsu)

        kernel_sm = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean     = cv2.morphologyEx(raw_leaf, cv2.MORPH_OPEN, kernel_sm)

        cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled  = np.zeros_like(clean)
        if cnts:
            for cnt in cnts:
                if cv2.contourArea(cnt) >= 500:
                    cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)

        kernel_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        return cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_lg)

    def _generate_variant_a_mask(self, image_bgr, leaf_bin):
        hsv    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        raw    = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        masked = cv2.bitwise_and(raw, raw, mask=leaf_bin)
        k      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(masked, cv2.MORPH_OPEN, k)

    def _filter_blb_components(self, varA_mask, leaf_bin, leaf_px):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(varA_mask, connectivity=8)
        out = np.zeros_like(varA_mask)
        rem_px = 0

        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            blob = (labels == i).astype(np.uint8)

            w_b, h_b = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            asp      = max(w_b, h_b) / (min(w_b, h_b) + 1e-5)

            if area < self.min_blob_px and asp < 1.5:
                rem_px += area
                continue

            out[blob > 0] = 255

        return out, rem_px

    def _get_paper_disease_rating(self, pct: float) -> tuple:
        """
        Exact rating table from research paper:
          0 : 0
          1 : >1 - 10
          3 : >11 - 30
          5 : >31 - 50
          7 : >51 - 75
          9 : >76 - 100
        """
        if pct == 0.0:
            return 0, "Rating 0 (0%)"
        elif pct <= 10.0:
            return 1, "Rating 1 (1-10%)"
        elif pct <= 30.0:
            return 3, "Rating 3 (11-30%)"
        elif pct <= 50.0:
            return 5, "Rating 5 (31-50%)"
        elif pct <= 75.0:
            return 7, "Rating 7 (51-75%)"
        else:
            return 9, "Rating 9 (76-100%)"

    def _empty_result(self, shape):
        z = np.zeros(shape, dtype=np.uint8)
        return {
            "leaf_mask": z, "raw_hsv_mask": z, "final_lesion_mask": z,
            "leaf_pixels": 0, "lesion_pixels": 0, "removed_pixels": 0,
            "severity_pct": 0.0, "disease_rating": 0, "rating_label": "Rating 0 (0%)"
        }
