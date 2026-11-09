# 🌾 Rice Disease Detection & Severity Estimation AI

> **Internship Project** — End-to-end AI system for rice disease detection and severity estimation using Deep Learning and Computer Vision.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen?style=flat-square)
![Diseases](https://img.shields.io/badge/Diseases-12-purple?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey?style=flat-square)

---

## What This Project Does

Upload a rice plant photo and the AI will tell you:
1. **Which disease** it is — 97% accuracy
2. **How severe** the disease is — IRRI Grade 0 to 9
3. **Visual overlay** showing exactly where the disease is on the leaf

---

## How It Works

### Stage 1 — Disease Classification
- Model: **EfficientNet-B0** (Transfer Learning from ImageNet — Google 2019)
- Architecture: Final layer changed from `Linear(1280 → 1000)` to `Linear(1280 → 12)`
- Training: 9,700+ images, 50 epochs, Adam optimizer, GPU
- Result: **97% Accuracy**

### Stage 2 — Severity Estimation (Custom Pipeline)
1. **ExG Formula** → `2G - R - B` → isolates green leaf pixels
2. **Multi-Layer Exclusion** → removes soil, sky, shadow, skin false positives
3. **HSV Detection** → finds disease-colored lesion areas per disease
4. **Severity %** = `Lesion Pixels / Leaf Pixels x 100`
5. Mapped to **IRRI Standard Evaluation Scale (0-9)**

---

## All 12 Diseases

| Disease | Severity | Method |
|---------|----------|--------|
| Bacterial Leaf Blight | ✅ Active | ExG + Multi-Layer Exclusion + HSV (1,160 images verified, mean 5.34%) |
| Brown Spot | ✅ Active | ExG + HSV |
| Rice Blast | ✅ Active | ExG + HSV — diamond-shaped spot detection |
| Narrow Brown Leaf Spot | ✅ Active | ExG + HSV |
| Sheath Blight | ✅ Active | ExG + HSV + RLH formula (Panda et al. 2023) |
| Grain Discoloration | ✅ Active | HSV color-based grading (Jinorose et al. 2010) |
| Healthy | ✅ 0% | Baseline — no disease |
| False Smut | Classification Only | Smut ball counting needed (Baite et al. 2020) |
| Root Knot | Classification Only | Underground root damage — not visible in photo |
| Stem Rot | Classification Only | Internal stem damage — not visible externally |
| Bakanae | Classification Only | Plant height-based scale (Volante et al. 2017) |
| Sheath Rot | Classification Only | Flag leaf sheath not reliably isolatable |

---

## IRRI Standard Severity Scale

| Grade | Area Affected | Level |
|-------|--------------|-------|
| 0 | 0% | Healthy |
| 1 | Less than 1% | Highly Resistant |
| 3 | 1 to 5% | Mild |
| 5 | 6 to 25% | Moderate |
| 7 | 26 to 50% | Severe |
| 9 | More than 50% | Critical |

---

## Run Locally

```bash
git clone https://github.com/hafsaashraf678/-rice-disease-detection-ai.git
cd -rice-disease-detection-ai
pip install -r requirements.txt
```

> Add your trained `best_model.pth` inside the `model/` folder

```bash
python flask_app.py
```

Open browser: `http://localhost:5000`

---

## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.13 | Core language |
| PyTorch | 2.13.0 | Deep learning framework |
| OpenCV | 5.0.0 | Image processing |
| Flask | 3.1.3 | Web application backend |
| NumPy | Latest | Array mathematics |
| EfficientNet-B0 | — | Classification backbone |

---

## Research Papers Applied

1. IRRI Standard Evaluation System for Rice (SES), 2013
2. Tan and Le (2019) — EfficientNet: Rethinking Model Scaling for CNNs (ICML 2019)
3. Mandal et al. (2023) — Rice Blast spectral severity (Frontiers in Plant Science)
4. Volante et al. (2017) — Bakanae disease 0-4 per-plant severity scale
5. Baite et al. (2020) — False Smut severity formula
6. Jinorose et al. (2010) — HSV color space for Grain Discoloration
7. Nalawade et al. (2020) — Sheath Rot IRRI grading scale
8. Panda et al. (2023) — Sheath Blight RLH formula
9. Roy and Kukreja (2026) — Dual-output CNN validation
10. Patil et al. (2022) — EfficientNet-B0 best accuracy for rice disease

---

## Author

**Hafsa Ashraf** — AI Engineer, Deep Learning Researcher, Computer Vision

Built during internship. All decisions are grounded in published research.
