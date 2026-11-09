"""
scripts/20_brown_spot_severity_estimator.py
=============================================
Standalone Brown Spot Severity Estimator Class (Strategy D V3).

IRRI RICE DISEASE SEVERITY GRADES (Brown Spot Specific Scale):
  • Grade 0 (0.00%)        : Healthy / No Lesion
  • Grade 1 (0.01 - 1.00%) : Very Low / Mild
  • Grade 3 (1.01 - 5.00%) : Moderate
  • Grade 5 (5.01 - 10.00%): Severe
  • Grade 7 (10.01- 25.00%): Very Severe
  • Grade 9 (> 25.00%)     : Extreme / Critical
"""

import cv2
import numpy as np


class BrownSpotSeverityEstimator:
    def __init__(self, dilation_px=20):
        self.hsv_lower = np.array([0, 30, 15], dtype=np.uint8)
        self.hsv_upper = np.array([38, 255, 180], dtype=np.uint8)

        self.dark_lower = np.array([0, 50, 15], dtype=np.uint8)
        self.dark_upper = np.array([38, 255, 140], dtype=np.uint8)

        self.dilation_px          = dilation_px
        self.min_blob_px          = 15
        self.max_dark_area_pct    = 0.05
        self.max_aspect_ratio     = 3.5
        self.min_circularity      = 0.15
        self.max_boundary_contact = 0.15
        self.min_solidity         = 0.35
        self.min_saturation       = 55.0
        self.max_value            = 135.0

    def estimate_severity(self, image_bgr: np.ndarray, leaf_mask: np.ndarray = None) -> dict:
        if leaf_mask is None:
            leaf_mask = self._generate_fallback_leaf_mask(image_bgr)

        leaf_bin = (leaf_mask > 0).astype(np.uint8)
        leaf_px  = int(np.sum(leaf_bin))

        if leaf_px == 0:
            return self._empty_result(image_bgr.shape[:2])

        varA_mask   = self._generate_variant_a_mask(image_bgr, leaf_bin)
        stratD_mask = self._filter_strategy_D(varA_mask, leaf_bin, leaf_px)
        final_mask, recovered_px = self._dark_core_recovery(
            image_bgr, leaf_bin, leaf_px, varA_mask, stratD_mask)

        lesion_px    = int(np.sum(final_mask > 0))
        severity_pct = round(lesion_px / float(leaf_px) * 100.0, 2)
        irri_grade, severity_label = self._get_irri_grade(severity_pct)

        return {
            "leaf_mask"          : leaf_mask,
            "raw_hsv_mask"       : varA_mask,
            "strategyD_mask"     : stratD_mask,
            "final_lesion_mask"  : final_mask,
            "leaf_pixels"        : leaf_px,
            "lesion_pixels"      : lesion_px,
            "recovered_pixels"   : recovered_px,
            "severity_pct"       : severity_pct,
            "irri_grade"         : irri_grade,
            "severity_label"     : severity_label,
        }

    def _generate_fallback_leaf_mask(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k)

    def _generate_variant_a_mask(self, image_bgr, leaf_bin):
        hsv    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        raw    = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        masked = cv2.bitwise_and(raw, raw, mask=leaf_bin)
        k      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean  = cv2.morphologyEx(masked, cv2.MORPH_OPEN, k)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
        out = np.zeros_like(clean)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= 4:
                out[labels == i] = 255
        return out

    def _filter_strategy_D(self, varA_mask, leaf_bin, leaf_px):
        k      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(leaf_bin, k, iterations=1)
        bnd    = leaf_bin - eroded

        n, labels, stats, _ = cv2.connectedComponentsWithStats(varA_mask, connectivity=8)
        out = np.zeros_like(varA_mask)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 4: continue
            blob = (labels == i).astype(np.uint8)
            ar   = area / float(leaf_px)
            bc   = int(np.sum((blob > 0) & (bnd > 0))) / float(area)
            cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: out[blob > 0] = 255; continue
            cnt      = cnts[0]
            w_b, h_b = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            extent   = area / float(w_b * h_b) if (w_b * h_b) > 0 else 1.0
            asp      = max(w_b, h_b) / (min(w_b, h_b) + 1e-5)
            hull_a   = cv2.contourArea(cv2.convexHull(cnt))
            sol      = area / hull_a if hull_a > 0 else 1.0

            remove = (ar >= 0.15 or
                      (bc >= 0.25 and (extent < 0.22 or asp > 3.5)) or
                      (extent < 0.18 and sol < 0.25 and ar >= 0.05))
            if not remove: out[blob > 0] = 255
        return out

    def _dark_core_recovery(self, image_bgr, leaf_bin, leaf_px, varA_mask, stratD_mask):
        hsv  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        k_sm = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(leaf_bin, k_sm, iterations=1)
        bnd_px = leaf_bin - eroded

        k_dil     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.dilation_px + 1, 2 * self.dilation_px + 1))
        prox_D    = cv2.dilate(stratD_mask, k_dil, iterations=1)
        prox_varA = cv2.dilate(varA_mask,   k_dil, iterations=1)
        dual_prox = cv2.bitwise_or(prox_D, prox_varA)
        dual_prox = cv2.bitwise_and(dual_prox, dual_prox, mask=leaf_bin)

        raw_dark    = cv2.inRange(hsv, self.dark_lower, self.dark_upper)
        masked_dark = cv2.bitwise_and(raw_dark, raw_dark, mask=leaf_bin)
        clean_dark  = cv2.morphologyEx(masked_dark, cv2.MORPH_OPEN, k_sm)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(clean_dark, connectivity=8)

        recovered_mask = np.zeros_like(clean_dark)
        recovered_px   = 0

        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_blob_px: continue
            if (area / float(leaf_px)) > self.max_dark_area_pct: continue

            blob = (labels == i).astype(np.uint8)

            cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            cnt  = cnts[0]
            peri = cv2.arcLength(cnt, True)
            circ = (4.0 * np.pi * area) / (peri ** 2) if peri > 0 else 1.0
            w_b, h_b = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            asp      = max(w_b, h_b) / (min(w_b, h_b) + 1e-5)
            bc_px    = int(np.sum((blob > 0) & (bnd_px > 0)))
            bc_rat   = bc_px / float(area)
            hull_a   = cv2.contourArea(cv2.convexHull(cnt))
            sol      = area / hull_a if hull_a > 0 else 1.0

            if circ < self.min_circularity: continue
            if asp  > self.max_aspect_ratio: continue
            if bc_rat > self.max_boundary_contact: continue
            if sol  < self.min_solidity: continue

            hsv_vals = hsv[blob > 0]
            s_mean   = float(np.mean(hsv_vals[:, 1]))
            v_mean   = float(np.mean(hsv_vals[:, 2]))
            if s_mean < self.min_saturation: continue
            if v_mean > self.max_value: continue

            overlap = int(np.sum((blob > 0) & (dual_prox > 0)))
            if overlap == 0: continue

            new_px = int(np.sum((blob > 0) & (stratD_mask == 0)))
            if new_px > 0:
                recovered_mask[blob > 0] = 255
                recovered_px += new_px

        final_mask = cv2.bitwise_or(stratD_mask, recovered_mask)
        return final_mask, recovered_px

    def _get_irri_grade(self, pct: float) -> tuple:
        if pct == 0.0:
            return 0, "Healthy / No Lesion"
        elif pct <= 1.0:
            return 1, "Very Low / Mild"
        elif pct <= 5.0:
            return 3, "Moderate"
        elif pct <= 10.0:
            return 5, "Severe"
        elif pct <= 25.0:
            return 7, "Very Severe"
        else:
            return 9, "Extreme / Critical"

    def _empty_result(self, shape):
        z = np.zeros(shape, dtype=np.uint8)
        return {
            "leaf_mask": z, "raw_hsv_mask": z, "strategyD_mask": z, "final_lesion_mask": z,
            "leaf_pixels": 0, "lesion_pixels": 0, "recovered_pixels": 0,
            "severity_pct": 0.0, "irri_grade": 0, "severity_label": "Healthy / No Lesion"
        }
