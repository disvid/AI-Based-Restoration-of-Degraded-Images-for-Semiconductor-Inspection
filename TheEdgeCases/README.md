# AI-Based Restoration of Degraded Images — KLA Problem Statement

**Team:** TheEdgeCases · Indian Institute of Technology, Roorkee
**Event:** Hackathon 2026, organized as part of SEMICON India

Joint denoising and 2× super-resolution of degraded grayscale semiconductor
inspection images. A single feed-forward network (**JDSR-NAF**, 13.73 M
parameters) removes multiplicative speckle and additive shot noise while
recovering full spatial resolution in one pass.

---

## 1. Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run.py <input-dir> <output-dir>
```

`run.py` is the submission entry point and takes two positional arguments.
It requires no internet access, API keys, model downloads, manual edits, or
interactive input. Model weights ship in `models/`. It runs on an NVIDIA GPU
when one is available and falls back to CPU automatically.

Example:

```bash
python run.py ./Test_NoisyLR ./test_predictions
```

## 2. Input / output contract

| | Specification |
|---|---|
| Input | `.npy`, grayscale, shape `(H, W)`; values may fall outside `[0, 1]` |
| Output | `.npy`, grayscale, shape `(2H, 2W)`, `float32`, clipped to `[0, 1]`, finite |
| Naming | Each output has exactly the same filename as its input |
| Output dir | Created automatically if it does not exist |

Clipping and `torch.nan_to_num` are applied inside `run.py`, so outputs are
scored exactly as saved. Any input side length is accepted — the network
reflect-pads internally to a multiple of 8 and crops back after upscaling.
Inputs of differing sizes in the same directory are handled: batches are
grouped by shape.

## 3. Repository contents

```
TheEdgeCases/
├── run.py                       # SUBMISSION ENTRY POINT — positional args
├── requirements.txt             # inference dependencies (pinned)
├── README.md                    # this file
├── models/
│   ├── best.pt                  # submitted checkpoint (perceptual phase)
│   └── fidelity_variant.pt      # ablation baseline (fidelity-only phase)
├── train.py                     # training entry point
├── jdsr_naf.py                  # network architecture
├── calibrated_degradation.py    # calibrated degradation simulator
├── calibrate_degradation.py     # degradation-recovery analysis tool
├── evaluate.py                  # scoring script (PSNR / SSIM / LPIPS)
├── requirements-train.txt       # full training environment
├── calibration.json             # recovered degradation parameters
└── results/                     # metrics, restored examples, failure cases
```

`run.py` is fully self-contained — it re-declares the architecture inline and
imports only `torch` and `numpy`. The remaining files are required for
reproducibility and are not touched during inference.

## 4. Method

### 4.1 Degradation calibration

Rather than assume degradation parameters, we recovered them from the 3,200
provided pairs using `calibrate_degradation.py`.

| Property | Measured value | Method |
|---|---|---|
| Scale factor | 2× (all 3,200 pairs are 256→128) | header survey |
| Downsampler | **torch bicubic, `antialias=False`, `align_corners=False`** (a = −0.75) | 12×12 least-squares kernel recovery; ties `cv2.INTER_CUBIC` to 7 s.f. |
| Noise variance | `Var(μ) = 0.0197 μ² + 0.0055 μ + ~0` (R² = 0.99941) | quantile-binned weighted least squares on residuals |
| Speckle | Gamma, **L ≈ 40** (σ/μ ≈ 16 %) | 1/a; MLE beat lognormal and normal (324827 vs 318335 vs 318264) |
| Additive term | signal-proportional (shot-like); **no constant Gaussian floor** | `Var/μ²` decays to a 0.025 asymptote |
| Noise ordering | speckle white at LR; additive component correlated by bicubic's negative side lobes | residual lag-1 autocorrelation −0.057 measured vs −0.154 × additive fraction predicted |
| Intensity range | NoisyLR ∈ [−0.220, 2.158]; 2.71 % of pixels > 1, 0.108 % < 0 | direct measurement |

The kernel recovery works despite a noisy regression target because mean-1
multiplicative noise satisfies `E[noisy] = clean`, leaving the least-squares
estimator unbiased. Fitted on 900,000 sampled LR pixels across 300 images.

Two findings drove the architecture:

1. **The downsampler is known exactly**, so its closed-form pseudo-inverse
   (bicubic upsample) is added as a gated global residual and the network
   predicts only the correction. The output convolution is zero-initialised,
   so the model emits exactly the bicubic baseline at step 0.
2. **Speckle is mild in relative terms (L ≈ 40)** — far milder than the
   L ∈ [1, 12] our first iteration assumed. That earlier assumption trained
   the network on noise 3–6× more severe than reality and taught it to
   over-smooth.

### 4.2 Architecture — JDSR-NAF (13.73 M parameters)

```
input (B,1,H,W)
  ├── bicubic ×2 ─────────────────────────────────┐  (global residual, learnable gate)
  └── SymlogStem: cat[x, sign(x)·log1p|x|] → 48ch  │
      ├── NAFBlock ×2  (3×3 depthwise)  ── skip ──┐│
      ├── NAFBlock ×2  (5×5 depthwise)  ── skip ─┐││
      ├── NAFBlock ×4  (7×7 depthwise)  ── skip ┐│││
      ├── MDTA TransformerBlock ×6, 8 heads     ││││
      ├── PixelShuffle decoder ×3 (ICNR init) ──┘┘┘│
      └── SRHead: sub-pixel ×2 → refine → 1ch ─────┴──→ clamp[0,1]
```

- **Symlog stem** rather than `log(x + ε)`: 0.108 % of input pixels are
  genuinely negative (min −0.22), which `log` maps to −13.8 — an outlier three
  orders of magnitude outside the rest of the channel. `sign(x)·log1p|x|` is
  smooth and finite on all of ℝ, exactly 0 at 0, and still compresses the
  multiplicative bright tail.
- **NAFNet blocks** (SimpleGate, activation-free) for encoder and decoder.
  Per-channel residual gates initialise to zero, so depth trains stably.
- **Depthwise kernels widen with depth** (3→5→7): a 7×7 depthwise convolution
  on a 16×16 bottleneck map is nearly free, while the same kernel at full
  resolution would dominate cost.
- **Restormer MDTA channel attention at the bottleneck only** — cost linear in
  spatial size, providing global context to distinguish genuine structure from
  speckle. Its interior runs in fp32 because `F.normalize` underflows in fp16.
- **ICNR initialisation** on every sub-pixel convolution, removing the
  checkerboard artefacts that LPIPS penalises heavily.
- **Raw single-channel input**; symlog is computed inside the model, so there
  is exactly one definition of the input transform and no train/inference skew.

Three presets (`tiny` ≈ 2.5 M, `base` = 13.73 M, `large` ≈ 26 M) provide a
quality–latency trade-off. The submitted model uses `base`.

### 4.3 Training

Two phases. AdamW (`betas = (0.9, 0.99)`), cosine schedule with warmup, fp16
AMP, `channels_last`, gradient clipping at 1.0, EMA decay 0.999. No weight
decay on 1-D parameters (norm affines, residual gates, attention temperature).

| | Phase 1 (fidelity) | Phase 2 (perceptual) |
|---|---|---|
| Loss | Charbonnier + 0.15·(1 − SSIM) | + 0.12·LPIPS |
| Learning rate | 4e-4, then 1e-4 (extension) | 4e-5 |
| Epochs | 30 + 20 | 15 |
| LR patch | 96 × 96 | 96 × 96 |
| Batch | 16 | 16 |
| Selection | best in-distribution composite | best OOD composite |

Charbonnier runs on the **unclamped** prediction so out-of-range excursions are
penalised; SSIM and LPIPS run on the clamped prediction, since both are only
meaningful on a bounded range. Both are forced to fp32 — SSIM's stability
epsilon lies below fp16's smallest normal value.

**Data pipeline.** Each sample is drawn either from the real provided pair
(50 %) or freshly synthesised from ground truth through the calibrated
simulator (50 %). Within the simulator, 78 % of draws use the exact measured
parameters and 22 % use deliberately widened ranges (L log-uniform in
[8, 140], alternate resamplers, occasional extra blur). Augmentation is D4
(all eight dihedral transforms), content-scale jitter, and spatial compositing
of two independently-degraded versions of the same ground truth (p = 0.08),
which teaches spatially adaptive restoration strength.

**CutMix was implemented, measured, and removed.** It cost roughly 0.5 dB:
dense per-pixel regression learns to blur across the sharp content
discontinuities it introduces, consistent with Yoo et al., *"Rethinking Data
Augmentation for Image Super-Resolution"* (CVPR 2020). All submitted runs use
`--p_cutmix 0.0`.

**Validation split.** 160 images (5 %) held out by seeded shuffle, used for
neither gradient updates nor synthesis, evaluated at full resolution rather
than on patches. A second 80-image out-of-distribution split is generated with
`p_in_dist = 0.0` and **cached once at construction**, so checkpoint selection
is not made against a moving target.

## 5. Results

Validation set: 160 held-out images, full resolution. LPIPS uses the AlexNet
backbone. Composite = `(PSNR/40 + SSIM + (1 − LPIPS)) / 3`.

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Composite |
|---|---:|---:|---:|---:|
| Bicubic upsample of noisy input (no learning) | 22.89 | — | — | — |
| Lightweight RepConv CNN, 6 blocks | 27.69 | — | — | — |
| JDSR-NAF, assumed degradation model | 28.140 | 0.7775 | 0.2377 | 0.7500 |
| JDSR-NAF, calibrated + fidelity phase | **28.735** | **0.7943** | 0.2377 | 0.7583 |
| **JDSR-NAF, perceptual phase — submitted** | 28.507 | 0.7865 | **0.1260** | **0.7911** |

Out-of-distribution split (80 images, widened degradation parameters):

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| JDSR-NAF, fidelity phase | 27.285 | 0.7394 | 0.3270 |
| **JDSR-NAF, perceptual phase — submitted** | **27.414** | **0.7477** | **0.1813** |

Two results are worth isolating:

- **+5.6 dB over the do-nothing bicubic baseline** (22.89 → 28.51).
- **+0.6 dB purely from calibrating the degradation model** rather than
  assuming it (28.140 → 28.735, rows 3 → 4, identical architecture).

The perceptual phase trades 0.23 dB PSNR for a 0.11 LPIPS improvement.
Since KLA combines all three metrics with undisclosed weights, this is
favourable under essentially any weighting; the fidelity checkpoint ships
alongside as `models/fidelity_variant.pt`.

**Independent verification.** Both checkpoints were re-scored from disk with
`evaluate.py` over all 3,200 images, loading EMA and raw weights separately:

| Checkpoint | Weights | PSNR | SSIM | LPIPS | Composite |
|---|---|---:|---:|---:|---:|
| fidelity | EMA | 28.5218 | 0.7855 | 0.2430 | 0.75185 |
| fidelity | raw | 28.5189 | 0.7856 | 0.2418 | 0.75225 |
| perceptual | EMA | 28.3049 | 0.7779 | 0.1290 | 0.78550 |
| perceptual | raw | 28.2995 | 0.7779 | 0.1284 | 0.78566 |

EMA and raw agree to within 0.006 dB, confirming full convergence. Over the
full set, held-out images score ~0.22 dB *higher* than trained-on images — no
measurable overfitting.

**Additional selection metric.** Checkpoints were chosen on the composite
above rather than PSNR alone, because PSNR-only selection systematically
prefers the over-smoothed checkpoint.

### 5.1 Error decomposition

| Reference | PSNR |
|---|---:|
| Bicubic upsample of noisy input (do nothing) | 22.89 dB |
| Bicubic upsample of cleanly downsampled GT (perfect denoising, no learned SR) | 31.56 dB |
| Noisy-upsampled vs clean-upsampled (noise contribution alone) | 24.07 dB |

Converting to MSE, noise accounts for roughly 76 % of baseline error and
super-resolution roughly 14 %. **Residual noise, not super-resolution
capability, is the current binding constraint** — the submitted model sits
~2.9 dB below the perfect-denoising reference. See §7.

### 5.2 Failure cases

Worst three validation images by PSNR (mean 28.5 dB); restored examples at
full resolution are in `results/`.

| File | PSNR |
|---|---:|
| `002973.npy` | 11.49 dB |
| `000407.npy` | 15.82 dB |
| `002534.npy` | 18.13 dB |

## 6. Runtime

| | |
|---|---|
| Hardware measured | NVIDIA Tesla T4 (Google Colab); target hardware is NVIDIA H100 |
| Parameters | 13.73 M |
| Batch size | 32, grouped by input shape |
| Precision | bf16 autocast where supported, fp16 fallback; `channels_last` |
| Timing method | `time.perf_counter()` in `run.py`, spanning read → transfer → inference → write |

| Run | Images | Model init | Total | Per image |
|---|---:|---:|---:|---:|
| **Test set, cold start** | 400 | 1.26 s | **22.39 s** | 52.8 ms |
| Training set, warm GPU | 3,200 | 0.63 s | 38.37 s | 11.8 ms |

The cold-start figure is the representative one for a single evaluator
invocation. The warm run benefits from an already-initialised CUDA context and
cached cuDNN algorithm selections. H100 figures are not reported because we
did not have access to that hardware; the fixed overheads (interpreter
startup, CUDA context creation) are hardware-independent and dominate.

Throughput decisions:

- **`torch.compile` is deliberately not used at inference.** Inductor warmup
  costs 30–90 s, far exceeding total inference time for a few hundred small
  images. It is available for training only.
- No `lpips`, `torchvision`, `matplotlib`, or `pandas` import anywhere in
  `run.py` — `lpips` alone triggers a 233 MB weight download, which would
  violate the no-internet requirement.
- Threaded parallel `.npy` reads and a background writer thread overlap disk
  I/O with GPU compute; `cudnn.benchmark` and TF32 are enabled.

## 7. Limitations and next steps

- **Residual noise is the binding constraint, not super-resolution.** The
  error decomposition in §5.1 shows the model sits ~2.9 dB below the
  perfect-denoising reference. The identified next step is additional capacity
  at the LR-resolution encoder level (more blocks and wider depthwise kernels
  at level 0), where speckle suppression happens.
- **No 512→256 training pairs exist.** Every provided pair is 256→128, despite
  the problem statement listing 512×512 ground truth. Content-scale jitter
  partially compensates by varying spatial frequency relative to the pixel
  grid, but performance on 512×512 targets is extrapolated rather than
  validated. Degrading an external high-resolution corpus through the
  calibrated pipeline is the clearest remaining improvement.
- **The model is fully converged at this capacity.** Train and validation
  metrics were flat to four decimal places over the final 16 epochs with no
  train/validation divergence. A larger preset and longer schedule are the
  obvious next experiment.
- **The additive noise component is imperfectly modelled.** Measured variance
  is best fit by a signal-proportional term rather than a constant Gaussian
  floor; the simulator approximates this by applying it at HR before
  downsampling. Residual autocorrelation matches (−0.057 measured vs −0.055
  predicted), but the true generating process is not disclosed.

## 8. External resources

| Resource | Use | Licence | Link |
|---|---|---|---|
| PyTorch | framework | BSD-3-Clause | https://pytorch.org |
| NumPy | array I/O | BSD-3-Clause | https://numpy.org |
| `lpips` (Zhang et al., CVPR 2018) | **training loss and evaluation only** | BSD-2-Clause | https://github.com/richzhang/PerceptualSimilarity |
| torchvision AlexNet, ImageNet weights | LPIPS backbone, **training only** | BSD-3-Clause | https://pytorch.org/vision |
| SciPy, Matplotlib, OpenCV | calibration analysis and plots only | BSD-3-Clause / PSF / Apache-2.0 | — |

**No external image datasets were used.** All training data derives from the
official KLA training set, either as the provided pairs or as synthetic pairs
generated from the provided ground truth. **The submitted inference path
(`run.py` + `models/best.pt`) has no pretrained-weight dependency and requires
no network access.**

Architectural components are reimplemented from published work; no
third-party model weights are loaded:

- Chen et al. (ECCV 2022) — *Simple Baselines for Image Restoration* (NAFNet):
  activation-free gated blocks and simplified channel attention.
- Zamir et al. (CVPR 2022) — *Restormer*: channel-wise self-attention with
  linear spatial complexity.
- Aitken et al. (2017) — ICNR initialisation for checkerboard-artefact-free
  sub-pixel convolution.
- Yoo et al. (CVPR 2020) — *Rethinking Data Augmentation for Image
  Super-Resolution*: basis for removing CutMix.

## 9. Reproducing the submitted checkpoint

```bash
pip install -r requirements-train.txt

# Recover degradation parameters from the provided pairs
python calibrate_degradation.py --gt_dir <GT_DIR> --noisy_dir <NOISY_DIR> \
    --out_json calibration.json

# Phase 1 — fidelity
python train.py --gt_dir <GT_DIR> --noisy_dir <NOISY_DIR> \
  --preset base --phase fidelity \
  --epochs 30 --repeat 1 --batch_size 16 --lr_patch 96 \
  --lr 4e-4 --min_lr_frac 0.05 --warmup_steps 300 \
  --p_cutmix 0.0 --p_mixed_degradation 0.08 --p_in_dist 0.78 \
  --w_ssim 0.15 --select_on id --val_every 2 \
  --out_dir ./runs/phase1_fidelity

# Phase 1 — continued at lower learning rate
python train.py --gt_dir <GT_DIR> --noisy_dir <NOISY_DIR> \
  --preset base --phase fidelity --init_from ./runs/phase1_fidelity/best.pt \
  --epochs 20 --repeat 1 --batch_size 16 --lr_patch 96 \
  --lr 1e-4 --min_lr_frac 0.05 --warmup_steps 100 \
  --p_cutmix 0.0 --p_mixed_degradation 0.08 --p_in_dist 0.78 \
  --w_ssim 0.15 --select_on id --val_every 2 \
  --out_dir ./runs/phase1_fidelity_extended

# Phase 2 — perceptual
python train.py --gt_dir <GT_DIR> --noisy_dir <NOISY_DIR> \
  --preset base --phase perceptual \
  --init_from ./runs/phase1_fidelity_extended/best.pt \
  --epochs 15 --repeat 2 --batch_size 16 --lr_patch 96 \
  --lr 4e-5 --min_lr_frac 0.1 --warmup_steps 150 \
  --p_cutmix 0.0 --p_mixed_degradation 0.08 --p_in_dist 0.78 \
  --w_ssim 0.15 --w_lpips 0.12 --lpips_start_frac 0.0 \
  --select_on ood --val_every 2 \
  --out_dir ./runs/phase2_perceptual
```

Scoring an existing checkpoint against ground truth:

```bash
python evaluate.py --input_dir <NOISY_DIR> --output_dir /tmp/out \
    --weights models/best.pt --gt_dir <GT_DIR> --metrics
```

Seed 42 throughout (`seed_everything`, plus per-worker seeding).
`cudnn.benchmark` is enabled for speed, so results reproduce to within
~0.01 dB rather than bit-exactly.

## 10. Contact

Sourav Gupta (team leader) · souravgupta09295@gmail.com
Repository: https://github.com/disvid/AI-Based-Restoration-of-Degraded-Images-for-Semiconductor-Inspection
