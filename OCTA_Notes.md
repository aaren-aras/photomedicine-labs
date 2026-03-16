# OCTA 🟡

**O**ptical **C**oherence **T**omography (👀 cross-sections) **A**ngiography (👀 **microvasculature** + **blood flow**); for retina and ==choroid== (middle layer providing blood supply and regulating temperature)
- "**Functional** imaging extension of structural OCT"
- Non-invasive, no dye injection required (unlike FA/ICGA)
- Detects flow by comparing **decorrelation** of repeated B-scans at same location

---

## OCT

- Uses ==low-coherence interferometry== with ==near-infrared (NIR, ~700–2500 nm)== light for high-resolution, ==**depth**-resolved imaging==; "coherence gating"
  - Low-coherence = predictable phase over only short distances → interference only occurs when path lengths are nearly equal → restricts signal to a precise depth
  - (📈 localized, 📉 fringes)

![[Interference Fringes.png|300]]

- Uses ==Michelson interferometer==: splits light into **reference** and **sample** arms, reflects off mirrors, recombines for ==constructive interference==

![[Michelson Interferometer.png|400]]

- Backscattered light from different tissue depths returns with distinct optical path length delays
  - Only light matching the reference arm path length **interferes constructively**
  - Analogy: *optical ultrasound* (depth-resolved like US, but uses light)
- Series of 1D ==A-scans== (**a**mplitude vs. depth) → stacked into 2D ==B-scan== (**b**rightness image) ^bf8227
- Low contrast between small blood vessels and retinal tissue → need angiographic extension ([[OCTA 🟡#^098a9c|other modalities]])

### OCT Variants

| Variant | Full Name | Key Feature |
|---------|-----------|-------------|
| SD-OCT | Spectral Domain | Spectrometer-based, ~70 kHz A-scan rate, standard clinical |
| SS-OCT | Swept Source | Tunable laser, ~100–400 kHz, deeper penetration, less sensitivity roll-off |
| FD-OCT | Fourier Domain | Umbrella term covering both SD and SS |

### Sensitivity Roll-off
The decrease in signal strength (SNR) as imaging **depth increases** in FD-OCT (both SD and SS).
- **Why it happens**: deeper structures generate higher-frequency interference fringes; finite spectral resolution of the detector cannot resolve them as accurately
- SD-OCT: more severe roll-off (~10 dB/mm), limits choroidal imaging
- SS-OCT: less roll-off due to narrower instantaneous linewidth of swept laser → better for deeper structures (choroid, sclera)
- Relevant to Andrei's setup: SS-OCTA systems have advantage for imaging chicken embryo vasculature at depth

---

## Angiography

Produce **angiograms** — maps of blood vessel structure and flow
- Traditional methods: catheter, ==FA== (fluorescein angiography), ==ICGA== (indocyanine green angiography), ==MRA== (MR angiography), ==CTA== (CT angiography) ^098a9c
  - Gold standard: 30° stereo FA with 2.5× magnification of macula/posterior pole
  - Except… OCT is already standard in ophthalmic workflows → [[OCTA 🟡#Using OCT(A)|can we use it here too?]]

### Using OCT(A)

Non-invasive technique evaluating changes in **laser backscattering of moving RBCs** between sequentially acquired B-scans
- Shorter acquisition than FA/ICGA
- No dye injection → no adverse reactions, repeatable

![[Pasted image 20260128171214.png|400]]

#### Motion Detection Methods

OCTA detects *change* between consecutive B-scans — stationary tissue = constant signal; moving RBCs = decorrelated signal

| Method | Basis | Pros | Cons |
|--------|-------|------|------|
| Amplitude decorrelation | Compares B-scan intensities | Simple, robust | Motion artifact sensitive |
| Phase variance | Detects phase shift from moving objects | Sensitive to slow flow | Requires phase-stable system |
| ==SSADA== | **S**plit-**s**pectrum **a**mplitude **d**ecorrelation **a**ngiography | Gold standard for retinal OCTA; splits spectrum to increase independent decorrelation measurements, improving SNR | Computationally heavier |
| Complex signal (ED-OCTA) | Eigendecomposition of complex OCT signal | Less sensitive to bulk motion; used in Liao et al. 2023 | Requires more repeats |

##### SSADA — Deep Dive
The standard motion detection algorithm in commercial OCTA devices (Zeiss, Optovue):
1. Split the OCT spectrum into sub-bands (e.g., 4 bands)
2. Compute amplitude decorrelation between repeat B-scans **within each sub-band**
3. Average decorrelation maps across sub-bands
- **Why splitting helps**: sub-band images have lower axial resolution but are statistically more independent → averaging reduces speckle noise by √N
- Result: higher SSADA contrast-to-noise ratio vs. full-spectrum decorrelation
- Used in OCTA-500 dataset generation

#### Averaging Methods (En Face Projections)

To produce 2D OCTA maps from 3D volumes, OCTA decorrelation signals are projected across depth slabs:
1. **Split-spectrum amplitude decorrelation**: SSADA (above)
2. **Volume averaging**: max or mean projection within a depth range

Four standard retinal slabs:
- Superficial retinal plexus (SRP)
- Deep retinal plexus (DRP)
- Outer retina (avascular — any signal here = artifact)
- Choriocapillaris

→ OCTA-500 uses **ILM–OPL slab** which captures SRP + DRP = the two main vascular layers we're segmenting

#### FAZ Extraction

==FAZ== (**F**oveal **A**vascular **Z**one): central retinal region devoid of blood vessels
- Quantified metrics: area (mm²), perimeter, circularity, fractal dimension
- Clinical relevance: enlarged/irregular FAZ indicates diabetic retinopathy, macular degeneration, sickle cell disease
- Automated via ML: OCTA-Net, VAFF-Net (see datasets below)
- Relevant to our pipeline: GT_FAZ labels in OCTA-500 allow FAZ segmentation as a secondary task

#### Disadvantages of OCTA

- Smaller FOV than FA (typically 3×3 or 6×6 mm vs. 30° FA field)
- Cannot detect very slow blood flow (below decorrelation threshold)
- **Projection artifacts**: signal from superficial vessels "leaks" into deeper slabs
- **Motion artifacts** ← main challenge, see below
- Segmentation errors at layer boundaries

#### Commercial Devices

- ZEISS Angioplex™
- Optovue AngioVue® (uses SSADA)
- OCTARA™
- TruTrack™ (Heidelberg, motion tracking)

---

## Motion Artifacts in OCTA 🚨

**This is the core problem Andrei's lab is working on.**

### Types of Motion Artifacts

| Artifact | Cause | Appearance | Frequency |
|----------|-------|-----------|-----------|
| Saccade stripe | Fast eye jump between repeat B-scans | Bright/dark **horizontal white stripe** across image | Common |
| Microsaccade | Tiny involuntary eye movements | Fine striping, subtle | Very common |
| Blink | Complete loss of signal | Dark horizontal band (full frame loss) | Occasional |
| Heartbeat/pulse | Axial pulsation of eye from cardiac cycle | Periodic distortion | Always present |
| Bulk motion | Large eye movement during volume | Misaligned B-scans, distorted vessel map | Common in clinic |

**Key physics**: because OCTA detects *change* between repeat B-scans, any motion between repeats — even microscopic — is interpreted as "flow signal" → white stripe artifacts indistinguishable from real vessels at first glance

### How Motion is Corrected (State of the Art)

#### Hardware Approaches
- **Eye tracking**: active beam steering to compensate for eye movement in real-time (Heidelberg TruTrack, Zeiss FastTrac)
- **Fixation targets**: reduce voluntary movement
- **Head stabilization**: chin/forehead rests
- **Faster acquisition**: SS-OCT at >200 kHz reduces inter-repeat time → less motion opportunity
  - Andrei's setup uses SS-OCT — this is the right choice for motion reduction

#### Software/Post-processing Approaches
1. **Rigid registration**: align B-scans using cross-correlation before decorrelation
2. **Affine/deformable registration**: handle non-rigid deformations
3. **Outlier rejection**: discard B-scans with anomalous intensity (blink detection)
4. **Averaging**: more repeat B-scans (NR) → SNR ∝ √NR, dilutes motion artifacts
   - Standard: 4–8 repeats for retinal OCTA
   - Tradeoff: more repeats = longer scan = more opportunity for motion 🔄

#### Deep Learning Approaches (Current Research Focus)

**This is what the two papers Andrei sent are about:**

| Paper | Method | Input → Output | Key Result |
|-------|--------|---------------|------------|
| Liao et al. 2023 (IRU-Net) | ResU-Net + VGG19 content loss | 2-repeat OCTA → 12-repeat quality | PSNR: 15.7→24.2, SSIM: 0.28→0.59, 83% scan time reduction |
| Das et al. 2025 (RRTGAN) | Transformer GAN | 1/4 pixel sampling → full resolution | PSNR highest, DISTS/LPIPS best among methods tested |

**Our pipeline implements Option C (restore → segment):**
- **Stage 1 (Restoration)**: simulate motion artifacts on clean OCTA-500 images → train ResU-Net to reverse them
  - Loss: L2 + 0.01 × VGG19 content loss (Liao et al. optimal configuration)
  - Metrics: PSNR ↑, SSIM ↑
- **Stage 2 (Segmentation)**: clean (or restored) OCTA → binary vessel mask
  - Loss: Dice + 0.3 × BinaryCrossentropy
  - Metrics: Dice ↑, IoU ↑, clDice ↑

### Motion Artifact Classification (Future Work)
Beyond removal, we could **classify** motion artifacts:
- Saccade vs. blink vs. bulk motion → different correction strategies
- Could use a lightweight classifier (ResNet-18 on individual B-scans) to tag artifact type before applying targeted correction
- Relevant to Andrei's chicken embryo data where motion sources are different (no eye movement, but heartbeat + involuntary organism motion)

---

## Datasets

| Dataset | Images | Labels | Resolution | Link |
|---------|--------|--------|-----------|------|
| **OCTA-500** ⭐ | 500 retinal OCTA volumes | LargeVessel, Artery, Vein, FAZ | 400×400 (3mm), 400×400 (6mm) | [IEEE DataPort](https://ieee-dataport.org/open-access/octa-500) |
| **ROSE** ⭐ | 117 retinal OCTA | Vessel (capillary level) | 304×304 | [Zenodo](https://zenodo.org/records/12775880) |
| **DERMA-OCTA** | Skin OCTA | Vessel | 512×512 | [Nature](https://www.nature.com/articles/s41597-025-05763-6) |
| Giarratano | Retinal OCTA | Vessel + FAZ | Various | [DataShare](https://datashare.ed.ac.uk/handle/10283/3528) |
| Nzakimuena | Retinal OCT + OCTA | — | Various | [Kaggle](https://www.kaggle.com/datasets/cnzakimuena/retinal-oct-octa-data) |

**Next step after OCTA-500**: train on ROSE (fine capillary detail) → then fine-tune on Andrei's chicken embryo data

### OCTA-500 Structure
```
OCTA-500/
    OCTA(ILM_OPL)/     ← en face projections, ILM to OPL slab (SRP + DRP)  ← WE USE THIS
    OCTA(OPL_BM)/      ← outer retina to BM (choriocapillaris) — avascular
    OCT/               ← structural B-scans
    Label/
        GT_LargeVessel/  ← binary large vessel masks  ← WE USE THIS
        GT_Artery/
        GT_Vein/
        GT_FAZ/
```

---

## Deep Learning for OCTA — Key Papers

| Paper | Task | Architecture | Result |
|-------|------|-------------|--------|
| Liao et al. 2023 | OCTA reconstruction (2→12 repeats) | IRU-Net (ResU-Net + dense blocks) | PSNR 24.2, SSIM 0.59 |
| Das et al. 2025 | Sparse pixel recovery (1/4→full) | RRTGAN (Transformer GAN) | Best PSNR/DISTS/LPIPS |
| Giarratano et al. 2020 | Vessel segmentation benchmark | U-Net variants | Dice ~0.80-0.85 |
| OCTA-Net | Multi-task vessel seg | CNN encoder-decoder | SoTA on ROSE |
| VAFF-Net | Vessel-Aware Feature Fusion | Attention U-Net | ROSE SoTA |

---

## Andrei's Chicken Embryo Data

**Current status**: raw B-scans and EVM frames — not yet angiograms

### What we have:
- `Bscans/`: structural OCT B-scans (cross-sectional, depth-resolved)
- `EVM_chickembryo_frames/`: likely Eulerian Video Magnification or raw OCT frames

### What we need to do:
1. **Compute OCTA signal**: use speckle variance or amplitude decorrelation between repeat B-scans to generate the angiogram from the raw B-scans
2. **3D Slicer visualization**: load computed OCTA volume, use Volume Rendering module for 3D angiogram
3. **Run our segmentation model**: once OCTA images are generated, run through restore → segment pipeline

### 3D Slicer Workflow (for Friday task):
1. Open 3D Slicer → `Add Data` → load B-scan sequence as image stack
2. `Modules → Volume Rendering` → enable, adjust threshold to highlight vessels
3. `Modules → Segment Editor` → manually segment a few vessel regions (optional)
4. For proper OCTA: need to first compute speckle variance map in Python (cv2), then load result into Slicer

### Chicken Embryo vs Retinal OCTA:
| Factor | Retina (OCTA-500) | Chicken Embryo |
|--------|------------------|----------------|
| Motion source | Eye saccades, heartbeat | Organism motion, heartbeat |
| Vessel scale | Capillaries (~5–10 μm) | Larger vessels visible |
| FOV | 3×3 or 6×6 mm | Custom (depends on setup) |
| Ground truth | Manual annotation | Need to create |
| Transfer learning | Pre-train here | Fine-tune here |

---

## Our Pipeline Summary

```
Raw OCTA (degraded/sparse)
    │
    ▼ Stage 1 — ResU-Net Restoration
    │   Input:  simulated degraded OCTA (motion artifacts, sparse B-scans)
    │   Output: restored clean OCTA
    │   Loss:   L2 + 0.01 × VGG19 content loss
    │   Metrics: PSNR ↑, SSIM ↑
    │   Ref:    Liao et al. 2023 (IRU-Net)
    │
    ▼ Stage 2 — ResU-Net Segmentation
        Input:  clean (or restored) OCTA projection
        Output: binary vessel mask
        Loss:   Dice + 0.3 × BinaryCrossentropy
        Metrics: Dice ↑, IoU ↑, clDice ↑
        Ref:    Giarratano et al. 2020, OCTA-Net
```

---

## TODO / Research Questions

- [ ] Andrei: confirm SS-OCT vs SD-OCT for in-lab setup (affects sensitivity roll-off analysis)
- [ ] Compute speckle variance OCTA from Bscans.zip → run through our pipeline
- [ ] 3D Slicer angiogram visualization from chicken embryo data
- [ ] After OCTA-500 training: fine-tune on ROSE (capillary level) for better sensitivity
- [ ] Motion artifact **classification** (saccade vs blink vs bulk) — future extension
- [ ] Oxygen saturation ground truth? (mentioned in notes) — separate from flow detection
- [ ] Optica presentation April — present pipeline results to Andrei

---

#### Tags
#ura #octa #deep-learning #vessel-segmentation #motion-artifacts

#### References
- https://eyewiki.org/Optical_Coherence_Tomography_Angiography
- https://www.ncbi.nlm.nih.gov/books/NBK563235/
- Liao et al. 2023 — https://doi.org/10.1364/BOE.486933
- Das et al. 2025 — https://doi.org/10.1038/s44387-025-00038-2
- Giarratano et al. 2020 — https://doi.org/10.1167/tvst.9.13.5
