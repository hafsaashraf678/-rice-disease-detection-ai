"""
scripts/76_healthy_severity_estimator.py
=========================================
PHASE 4: Healthy Class — Standalone Severity Estimator Class

Always assigns 0.00% severity and Grade 0 (Healthy).
"""

import cv2
import numpy as np


class HealthySeverityEstimator:

    def estimate_severity(self, image_bgr: np.ndarray, mask: np.ndarray = None) -> dict:
        h, w = image_bgr.shape[:2]
        z    = np.zeros((h, w), dtype=np.uint8)
        tot  = h * w
        return {
            "mask"              : z,
            "raw_hsv_mask"      : z,
            "final_lesion_mask" : z,
            "total_pixels"      : tot,
            "lesion_pixels"     : 0,
            "removed_pixels"    : 0,
            "severity_pct"      : 0.0,
            "grade"             : 0,
            "severity_label"    : "Grade 0 (0% — Healthy)",
        }
