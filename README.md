# 📘 Teacher–Student Feature Regression for Unsupervised Defect Detection
### _Unsupervised anomaly detection using feature-space consistency and local regression_

<p align="center">
  <img src="examples/anomaly_map_example.png" width="500">
</p>

This repository implements a **fast, data-efficient, fully unsupervised defect-detection pipeline** for semiconductor wafer inspection.  
The method combines:

- **Sub-pixel image alignment**
- **Multi-scale deep teacher features (ResNet-18)**
- **Local patch-wise student regression**
- **R²-based defect likelihood maps**
- **Morphological cleanup for crisp detections**

The result is a method capable of detecting subtle defects that **pixel-wise differences fail to identify**.

---

# 🚀 Key Idea

Normal reference and inspected images share the same **manifold in feature space**.  
Defects **break this manifold consistency**.

To capture this, we:

1. Extract deep feature maps from a frozen **teacher** network (ResNet-18)  
2. For every pixel (x, y), extract a local feature patch  
3. Train a lightweight **student** model (linear regression or shallow MLP) to predict inspected features from reference features  
4. Compute **per-pixel R² scores** to quantify local consistency  
5. Define the **defect probability** as:  
   ```
   D = 1 - R²
   ```
   (Derived from the goodness-of-fit formulation in the mathematical model.)

Regions where the student fails to reconstruct inspected features correspond to **true defects**.

---

# 🧠 Method Overview

## 1. Alignment
Aligned images are essential for a valid feature comparison.  
We apply **phase cross-correlation** to achieve **sub-pixel alignment**.

## 2. Deep Feature Extraction  
We use **post-pooling feature maps** from ResNet-18 layers 1 and 2:
- Layer 1 → 64 channels  
- Layer 2 → 128 channels  

These layers gave the best balance of spatial detail and robustness.

## 3. Student Regression  
For each layer ℓ and pixel (x,y):

```
F_i_hat = W · vec(F_r) + b
```

Fitted with **Ridge regression (α = 1e−3)** to stabilize correlated feature dimensions.

## 4. Defect Likelihood (R² Map)

```
R² = 1 - (sum (F_i - F_i_hat)²) / (sum (F_i - mean(F_i))²)
```

Defect probability per layer:

```
D = 1 - R²
```

Final defect map = average across layers.

## 5. Thresholding & Morphology  
We apply:

- **Otsu thresholding**
- **Morphological open/close**  

This yields clean, interpretable detection masks.

---

# 📊 Results

### Pixel-wise differences fail  
Simple difference maps miss subtle defects.

### Teacher–Student regression succeeds  
The method detects:

- Small bright defects  
- Structural anomalies  
- Local pattern inconsistencies  

It remains robust across lighting changes due to feature-space modeling.

---

# 🧱 Repository Structure

```
muze-anomaly-detection/
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── teacher.py
│   │   └── student.py
│   ├── utils/
│   │   ├── patch_extraction.py
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   └── config.py
│   ├── train.py
│   ├── inference.py
│   └── evaluate.py
│
├── notebooks/
├── examples/
└── data/ (ignored)
```

---

# ⚙️ Installation

```bash
git clone https://github.com/lsusman/muze-anomaly-detection.git
cd muze-anomaly-detection
pip install -r requirements.txt
```

---

# 🏋️ Training

```bash
python src/train.py   --ref data/raw/case1_aligned_reference.png   --insp data/raw/case1_aligned_inspected.png   --save student_model.pth
```

Uses random brightness augmentation to expand the effective dataset.

---

# 🔍 Inference

```bash
python src/inference.py   --ref examples/ref_sample.png   --insp examples/insp_sample.png   --model student_model.pth
```

Produces:

- R² heatmap  
- Defect probability map  
- Final binary detection mask  
- Blob-level detections  

---

# 🧪 Why This Works

- Teacher features encode **multi-scale structure**  
- Student regression enforces **manifold consistency**  
- R² failure is a **stable anomaly signal**  
- Lightweight & interpretable  
- Works with **only one non-defective pair**  

---

# 📈 Limitations & Future Work

- Post-pooling features decrease spatial resolution  
- Detected regions may appear larger than the true defect  
- Can be improved by:
  - using pre-pooling features  
  - multi-resolution fusion  
  - non-linear student models  

---

# 🙌 Acknowledgments

Based on the accompanying summary report and method exploration.

---

# 🎯 Want enhancements?

I can add:

- pipeline diagrams  
- animated heatmaps  
- badges  
- Citation entries  
- A visual gallery of detections  

Just ask!
