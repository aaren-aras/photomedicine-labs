# Optical Coherence Tomography Angiography (OCTA)

**OCT** — cross-sectional structural imaging of the eye
**OCTA** — extends OCT to image microvasculature and blood flow in the retina and choroid (the middle layer responsible for blood supply and temperature regulation)

> "Functional imaging extension of structural OCT"

---

## OCT

Uses **low-coherence interferometry** with near-infrared (NIR, ~700–2500 nm) light for high-resolution, depth-resolved imaging. Low coherence means the light has a predictable phase over only very short distances, which restricts interference to a small range: a technique called **coherence gating** (maximises localisation, minimises spurious fringes).

Uses a **Michelson interferometer** to split light into reference and sample arms, reflect them off mirrors, and recombine them for constructive interference. Backscattered light from different tissue depths returns with distinct optical path length (time) delays; only light matching the reference arm's travel distance produces interference.

Conceptually similar to **optical ultrasound**.

- Series of 1D **A-scans** (amplitude) -> single 2D **B-scan** (brightness)
- Notably low contrast between small blood vessels and surrounding retinal tissue, which motivated the development of complementary modalities

---

## Conventional Angiography

Angiograms can be produced via:

- Catheter-based dye injection: **Fluorescein Angiography (FA)**, **Indocyanine Green Angiography (ICGA)**
- **MRI**: Magnetic Resonance Angiography (MRA)
- **X-ray**: Computed Tomography Angiography (CTA)

**Gold standard**: 30° stereo field of the macula and posterior pole with FA, at 2.5× magnification.

---

## OCTA

A **non-invasive** technique that evaluates changes in laser light reflectance/backscattering from moving red blood cells (RBCs) across sequentially acquired OCT images.

- Shorter acquisition time than conventional angiography
- Multiple A-scans -> B-scan, enabling detection of high flow rates

### Motion Detection Methods

-  **Amplitude decorrelation**: compares signal amplitudes between successive OCT B-scans
-  **Phase variance**: detects phase shifts caused by moving objects (e.g. RBCs)
- **Speckle variance**: detects intensity pattern changes by moving objects

### Averaging & Reconstruction

Images are segmented into four zones:
1. Superficial retinal plexus
2. Deep retinal plexus
3. Outer retina
4. Choriocapillaris

Key averaging approaches include:
- **Split-spectrum amplitude decorrelation (SSADA)**: operates on real/amplitude signal; better SNR
- **Volume averaging**

### Disadvantages

- Smaller field of view (FOV) than FA
- May not detect slow blood flow
- Projection artifacts
- Segmentation errors

---

## Commercially Available Devices

| Device | Manufacturer |
|---|---|
| Angioplex™ | ZEISS |
| AngioVue® | Optovue |
| OCTARA™ | — |
| TruTrack™ | — |

---

## SS-OCTA vs SD-OCTA

**Swept-Source OCTA (SS-OCTA)**
- Faster scan speeds (>100 kHz)
- ~1050 nm wavelength -> better tissue penetration
- Fewer motion artifacts

**Spectral-Domain OCTA (SD-OCTA)**
- Larger field of view
- Better visualisation of deeper retinal layers and choroidal vasculature

---

## Key Concepts

### Motion Artifacts
Because OCTA detects *change* (decorrelation) between consecutive B-scans, any eye movement (blinking, pulsation) is misread as blood flow, producing motion artifacts (typically white horizontal striping or distorted vessel maps).

### Sensitivity Roll-Off
The decrease in SNR as imaging depth increases in Fourier-domain OCT (FD-OCT), including both SD-OCT and SS-OCT. Deeper structures generate higher-frequency interference fringes that are harder to resolve due to finite spectral resolution.

### FAZ Extraction (Foveal Avascular Zone)
An image processing technique (usually applied to OCTA) that isolates and measures the central retinal region lacking blood vessels. Quantifies metrics such as area, perimeter, and circularity to aid diagnosis of retinal vascular diseases (e.g. diabetic retinopathy) using automated, ML-based algorithms.

---

## ED-OCTA Algorithm

The **Eigen-Decomposition OCTA (ED-OCTA)** algorithm is a complex-signal-based method used to visualise microvasculature by separating moving blood cells from static tissue. It functions as an **adaptive high-pass filter**.

Equation 1 decomposes the matrix of repeat B-scan signals at each pixel into eigenvectors sorted by eigenvalue magnitude:
- **Large eigenvectors** -> static tissue signal (high correlation across repeats)
- **Small eigenvectors** -> moving blood signal (low correlation, high decorrelation)

**Comparison with SSADA:**
- SSADA operates on real/amplitude signal -> better SNR, better for evaluation
- ED-OCTA uses phase shift (complex signal) -> better for blood flow detection, lower SNR

---

## Liao et al. — Deep Learning Enhancement of OCTA

### Motivation
Decorrelated signal quality improves as √N_R (where N_R = number of repeats), but each additional repeat increases scan time by ~1.75×. *Can we reconstruct 12-repeat quality OCTA from only 2-repeat acquisitions?*

### Dataset & Training Setup

- **Dataset:** 1784 image pairs from 21 raw OCT volumes; 77/23% train/val split
- **Ground truth:** all 12 repeats -> ED-OCTA -> high-quality OCTA image
- **Input:** first 2 of those 12 repeats -> ED-OCTA -> low-quality OCTA image
- Input resized from 600×600 -> **512×512**, normalised to [0, 1]
- Gaussian noise (σ = 0.4) added to inputs during training for generalisation

**Optimizer:** Adam — lr = 1×10⁻⁴, β₁ = 0.8, β₂ = 0.999
**LR decay:** ×0.95 every 10,000 steps
**Batch size:** 4 | **Epochs:** 400 (early stopping, patience = 20)
**Hardware:** NVIDIA RTX 3090

### IRU-Net Architecture

Standard **U-Net** encoder-decoder backbone with:

- **Modified Residual Dense Blocks (mRDB)**
  - Dense connections: every conv layer receives feature maps from all previous layers in the block (DenseNet concept)
  - No BatchNorm
  - LeakyReLU (prevents dead neurons)
  - No bias in conv layers
  - Kernel init: mean = 0, std = 0.02
  - Final layer uses 3×3 kernel (vs original RDB's 1×1)
- Both the mRDB output **and** the concatenated skip connections feed into the decoder

**Combined loss function:**

$$L_c = \alpha \cdot L_2 + \beta \cdot L_{content}$$

### Results

| Metric | 2-repeat (input) | 12-repeat (IRU-Net) | Improvement |
|---|---|---|---|
| PSNR | 15.7 dB | 24.2 dB | +8.5 dB |
| SSIM | 0.28 | 0.59 | +0.31 |
| Acquisition time | 21 s | 3.5 s | −83% |

Standard U-Net alone achieves PSNR 23.35 dB / SSIM 0.45. IRU-Net's gains come from the combination of mRDB dense connections *and* the content loss; neither alone is sufficient.

### Clinical Note on Hallucination Risk
The paper explicitly warns that shallow-layer results (Figure 10B) show more vascular detail than deep-layer results (Figure 10F), but cannot confirm whether the extra vessels are real or hallucinated. For clinical use, **deep-layer outputs are safer and more trustworthy**.

### Practical Takeaway
83% acquisition time reduction with validated PSNR/SSIM improvement. Implementation path: acquire paired training data on existing system -> train IRU-Net -> deploy on fast acquisitions. **No new hardware required.**

---

## Hardware Note: Galvanometer Scanners

In OCT systems, a **galvanometer-based scanner** steers the laser beam across the sample (e.g. retina) in two dimensions (X and Y) using motorized, high-precision mirrors. This enables the rapid, linear raster patterns required for B-scan acquisition and OCTA.

---

## References

- EyeWiki — [Optical Coherence Tomography Angiography](https://eyewiki.org/Optical_Coherence_Tomography_Angiography)
- NCBI Bookshelf — [OCT overview](https://www.ncbi.nlm.nih.gov/books/NBK563235/)
- Liao et al. — [AI-assisted dense OCTA imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC12689425/)

---
