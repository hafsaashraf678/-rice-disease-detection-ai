"""
scripts/31_grain_discoloration_severity_estimator.py
=====================================================
PHASE 4: Grain Discoloration — Standalone Severity Estimator Class

PDI-BASED RATING SCALE (Research Literature):
  Grade 0 : PDI = 0%        → Healthy (no discoloration)
  Grade 1 : PDI = 1–25%     → Slight
  Grade 2 : PDI = 26–50%    → Moderate
  Grade 3 : PDI > 50%       → Severe

DETECTION APPROACH:
  Step 1: Generate Grain Mask  (isolate grain from green/dark background)
  Step 2: Detect Discoloration (dark-brown patches within grain only)
  Step 3: Filter Components    (remove noise, keep real discoloration patches)
  Step 4: Compute PDI          (discolored_px / grain_px * 100)
  Step 5: Assign Grade         (0 / 1 / 2 / 3)
"""

import cv2
import numpy as np


class GrainDiscolorationSeverityEstimator:

    def __init__(self):
        # ── Grain mask HSV range (golden/straw grain color) ───────────────────
        self.grain_lower = np.array([8,  15,  60],  dtype=np.uint8)
        self.grain_upper = np.array([45, 230, 255], dtype=np.uint8)
        self.green_lower = np.array([35, 40,  40],  dtype=np.uint8)
        self.green_upper = np.array([85, 255, 255], dtype=np.uint8)

        # ── Discoloration detection HSV (dark-brown patches on grain) ─────────
        # Variant A (locked after Phase 2 tuning)
        self.disco_lower = np.array([0,  40,  20], dtype=np.uint8)
        self.disco_upper = np.array([25, 255, 120], dtype=np.uint8)

        # ── Component filter thresholds ───────────────────────────────────────
        self.min_blob_px    = 20    # ignore specks smaller than this
        self.max_grain_frac = 0.90  # ignore blobs > 90% of grain area

    # ─────────────────────────────────────────────────────────────────────────
    def estimate_severity(self, image_bgr: np.ndarray) -> dict:
        """
        Main entry point.
        Returns dict with:
          grain_mask, disco_raw_mask, disco_final_mask,
          grain_pixels, disco_pixels, pdi_pct, grade, grade_label
        """
        grain_mask = self._generate_grain_mask(image_bgr)
        grain_bin  = (grain_mask > 0).astype(np.uint8)
        grain_px   = int(np.sum(grain_bin))

        if grain_px == 0:
            return self._empty_result(image_bgr.shape[:2])

        raw_mask            = self._detect_discoloration(image_bgr, grain_bin)
        final_mask, rem_px  = self._filter_components(raw_mask, grain_bin, grain_px)

        disco_px   = int(np.sum(final_mask > 0))
        pdi_pct    = round(disco_px / float(grain_px) * 100.0, 2)
        grade, grade_label = self._get_grade(pdi_pct)

        return {
            "grain_mask"       : grain_mask,
            "disco_raw_mask"   : raw_mask,
            "disco_final_mask" : final_mask,
            "grain_pixels"     : grain_px,
            "disco_pixels"     : disco_px,
            "removed_pixels"   : rem_px,
            "pdi_pct"          : pdi_pct,
            "grade"            : grade,
            "grade_label"      : grade_label,
        }

    # ─────────────────────────────────────────────────────────────────────────
    def _generate_grain_mask(self, image_bgr):
        hsv        = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        grain_raw  = cv2.inRange(hsv, self.grain_lower, self.grain_upper)
        green_raw  = cv2.inRange(hsv, self.green_lower, self.green_upper)
        no_green   = cv2.bitwise_and(grain_raw, cv2.bitwise_not(green_raw))

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed  = cv2.morphologyEx(no_green, cv2.MORPH_CLOSE, k_close)
        opened  = cv2.morphologyEx(closed,   cv2.MORPH_OPEN,  k_open)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        total_px = image_bgr.shape[0] * image_bgr.shape[1]
        out      = np.zeros_like(opened)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= max(200, total_px * 0.005):
                out[labels == i] = 255
        return out

    def _detect_discoloration(self, image_bgr, grain_bin):
        hsv    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        raw    = cv2.inRange(hsv, self.disco_lower, self.disco_upper)
        within = cv2.bitwise_and(raw, raw, mask=grain_bin)
        k      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(within, cv2.MORPH_OPEN, k)

    def _filter_components(self, raw_mask, grain_bin, grain_px):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
        out     = np.zeros_like(raw_mask)
        rem_px  = 0
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_blob_px:
                rem_px += area
                continue
            if grain_px > 0 and (area / float(grain_px)) > self.max_grain_frac:
                rem_px += area
                continue
            out[labels == i] = 255
        return out, rem_px

    def _get_grade(self, pdi: float):
        if pdi == 0.0:
            return 0, "Grade 0 — Healthy (0%)"
        elif pdi <= 25.0:
            return 1, "Grade 1 — Slight (1-25%)"
        elif pdi <= 50.0:
            return 2, "Grade 2 — Moderate (26-50%)"
        else:
            return 3, "Grade 3 — Severe (>50%)"

    def _empty_result(self, shape):
        z = np.zeros(shape, dtype=np.uint8)
        return {
            "grain_mask": z, "disco_raw_mask": z, "disco_final_mask": z,
            "grain_pixels": 0, "disco_pixels": 0, "removed_pixels": 0,
            "pdi_pct": 0.0, "grade": 0, "grade_label": "Grade 0 — Healthy (0%)"
        }
