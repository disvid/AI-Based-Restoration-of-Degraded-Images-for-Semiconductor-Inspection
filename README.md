# AI-Based Restoration of Degraded Images - KLA Problem Statement

**Team:** `TheEdgeCases` · **Event:** Hackathon 2026, SEMICON India

Joint denoising and 2× super-resolution of degraded grayscale inspection
images. A single feed-forward network (**JDSR-NAF**, 13.73 M parameters)
removes multiplicative speckle and additive noise while recovering full
spatial resolution in one pass.

---

## 1. Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run.py <input-dir> <output-dir>
```

No internet access, API keys, model downloads, manual edits, or interactive
input are required. Model weights ship in `models/`. The script runs on an
NVIDIA GPU when available and falls back to CPU automatically.

## 2. Input / output contract

| | Specification |
|---|---|
| Input | `.npy`, grayscale, shape `(H, W)`; values may fall outside `[0, 1]` |
| Output | `.npy`, grayscale, shape `(2H, 2W)`, `float32`, clipped to `[0, 1]`, finite |
| Naming | Each output has exactly the same filename as its input |
| Output dir | Created if absent |

Clipping and `nan_to_num` are applied inside `run.py`, so outputs are scored
exactly as saved. Any input side length is accepted — the network reflect-pads
internally to a multiple of 8 and crops back after upscaling.

## 3. Repository layout
                         ┌──────────────────────┐
                         │      SUBMISSION      │
                         │        ENTRY         │
                         │       run.py         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────┐
                    │        MODEL CHECKPOINT     │
                    │                             │
                    │  models/best.pt             │
                    │  models/fidelity_variant.pt │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────────┐
              │              INFERENCE PIPELINE         │
              │                                        │
              │  src/jdsr_naf.py                       │
              │       │                                │
              │       ▼                                │
              │  src/calibrated_degradation.py         │
              │       │                                │
              │       ▼                                │
              │  src/evaluate.py                       │
              │       │                                │
              │       ▼                                │
              │  PSNR ── SSIM ── LPIPS                │
              └────────────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │         RESULTS          │
                    │                          │
                    │  final_metrics.json      │
                    │  results.md              │
                    │  examples/               │
                    │   ├─ success cases      │
                    │   └─ failure cases      │
                    └──────────────────────────┘


        ┌─────────────────────────────────────────────┐
        │              TRAINING PIPELINE               │
        └──────────────────────┬──────────────────────┘
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
   ┌──────────────────┐                ┌────────────────────┐
   │ train.py         │                │ Calibration Tool   │
   │ Training Entry   │                │                    │
   └────────┬─────────┘                │ calibrate_         │
            │                          │ degradation.py     │
            ▼                          └─────────┬──────────┘
   ┌──────────────────┐                          │
   │ kla_train.py     │◄─────────────────────────┘
   │                  │
   │ Dataset          │
   │ Metrics          │
   │ Losses           │
   │ Training Loop    │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────────────┐
   │ calibrated_degradation.py│
   │                          │
   │ Degradation Simulator    │
   └────────────┬─────────────┘
                │
                ▼
        ┌───────────────┐
        │ jdsr_naf.py   │
        │ JDSR-NAF      │
        │ Architecture  │
        └───────┬───────┘
                │
                ▼
        ┌────────────────┐
        │ best.pt        │
        │ Model Checkpoint│
        └────────────────┘


       ┌───────────────────────────────────────────┐
       │               CONFIGURATION               │
       │                                           │
       │ configs/requirements-train.txt            │
       │ configs/calibration.json                  │
       │                                           │
       │ requirements.txt → inference environment  │
       └───────────────────────────────────────────┘

       README.md
       → Documentation / Reproducibility


## 4. Method

### 4.1 Degradation calibration (this drove every design decision)

Rather than assume degradation parameters, we recovered them from the 3 200
provided pairs (`src/calibrate_degradation.py`):

| Property | Measured value | How |
|---|---|---|
| Scale factor | 2× (all 3 200 pairs are 256→128) | header survey |
| Downsampler | **torch bicubic, `antialias=False`, `align_corners=False`** (a = −0.75) | 12×12 least-squares kernel recovery; ties `cv2.INTER_CUBIC` to 7 s.f. |
| Noise variance | `var(μ) = 0.0197 μ² + 0.0055 μ + ~0` (R² = 0.99941) | quantile-binned WLS on residuals |
| Speckle | Gamma, **L ≈ 40** (σ/μ ≈ 16 %) | 1/a; MLE beat lognormal and normal |
| Extra term | signal-proportional (shot-like), **no constant Gaussian floor** | `var/μ²` decays to a 0.025 asymptote |
| Noise ordering | speckle white at LR; additive component correlated by bicubic's negative lobes | residual lag-1 autocorrelation −0.057 vs −0.154 theoretical |
| Intensity range | NoisyLR ∈ [−0.220, 2.158]; 2.71 % of pixels > 1, 0.108 % < 0 | direct measurement |

Two consequences shaped the model:

1. **The downsampler is known exactly**, so its closed-form pseudo-inverse
   (bicubic upsample) is added as a global residual and the network predicts
   only the correction. The output convolution is zero-initialised, so the
   model emits exactly the bicubic baseline at step 0.
2. **Speckle is mild (L ≈ 40, not L ≈ 1–12 as first assumed)**, so this is
   predominantly a super-resolution problem. Capacity and receptive field were
   prioritised over aggressive noise suppression.

### 4.2 Architecture — JDSR-NAF (13.73 M parameters)
input (B,1,H,W)
├── bicubic ×2 ──────────────────────────────────┐ (global residual, gated)
└── SymlogStem: [x, sign(x)·log1p|x|] → 48ch │
├── NAFBlock ×2 (3×3 depthwise) ── skip ──┐ │
├── NAFBlock ×2 (5×5 depthwise) ── skip ─┐│ │
├── NAFBlock ×4 (7×7 depthwise) ── skip ┐││ │
├── MDTA TransformerBlock ×6, 8 heads │││ │ (channel attention)
├── PixelShuffle decoder ×3 (ICNR init) ──┘┘┘ │
└── SRHead: sub-pixel ×2 → refine → 1ch ──────┴──→ clamp[0,1]


Design rationale:

- **Symlog stem** instead of `log(x + ε)`: 0.108 % of input pixels are
  genuinely negative (min −0.22), which `log` maps to −13.8 — a catastrophic
  outlier. `sign(x)·log1p|x|` is smooth and finite on all of ℝ, compresses the
  multiplicative bright tail, and is exactly 0 at 0.
- **NAFNet blocks** (SimpleGate, no activations) for the encoder/decoder:
  highly kernel-efficient, and the residual gates initialise to zero so depth
  trains stably.
- **Depthwise kernels widen with depth** (3→5→7): a 7×7 depthwise conv on a
  16×16 bottleneck map is nearly free, while the same kernel at full resolution
  would dominate cost.
- **MDTA channel attention at the bottleneck only**: cost is linear in spatial
  size, and it provides the global context needed to distinguish a genuine
  bright feature from a speckle spike. Its interior is forced to fp32 because
  `F.normalize` underflows in fp16.
- **ICNR initialisation** on every sub-pixel convolution, removing the
  checkerboard artefacts that LPIPS penalises heavily.

### 4.3 Training

Two phases, both AdamW (`betas=(0.9, 0.99)`), cosine schedule with warmup,
fp16 AMP, `channels_last`, gradient clipping at 1.0, EMA decay 0.999.

| | Phase 1 (fidelity) | Phase 2 (perceptual) |
|---|---|---|
| Loss | Charbonnier + 0.15·(1−SSIM) | + 0.12·LPIPS |
| LR | 4e-4 → 1e-4 (extended) | 4e-5 |
| Epochs | 30 + 20 | 15 |
| Patch (LR) | 96×96 | 96×96 |
| Batch | 16 | 16 |
| Selection | best in-distribution composite | best OOD composite |

**Data pipeline.** Each sample is drawn either from the real provided pair
(50 %) or freshly synthesised from ground truth through the calibrated
simulator (50 %). Within the simulator, 78 % of draws use the exact measured
parameters and 22 % use deliberately widened ranges (L ∈ [8, 140] log-uniform,
alternate resamplers, occasional extra blur). Augmentation is D4 (all eight
dihedral transforms) plus content-scale jitter, which varies spatial frequency
relative to the pixel grid — the axis along which 256→128 and 512→256 actually
differ, since no 512→256 pair exists in the training data.

CutMix and Mixup were **tested and removed**: they degrade dense regression
tasks (Yoo et al., *Rethinking Data Augmentation for Image Super-Resolution*,
CVPR 2020), costing roughly 0.5 dB in our runs.

**Validation split.** 160 images (5 %) held out by seeded shuffle, used for
neither gradient updates nor synthesis. A second 80-image OOD split is
generated with `p_in_dist=0.0` (always widened parameters) and cached once, so
selection is not made against a moving target.

## 5. Results

Validation set: 160 held-out images, full resolution, no patching.
LPIPS uses the AlexNet backbone.

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| Bicubic upsample of noisy input (no learning) | `<TODO>` | `<TODO>` | `<TODO>` |
| Earlier baseline: lightweight RepConv CNN, 6 blocks | 27.69 | — | — |
| JDSR-NAF, fidelity phase (`fidelity_variant.pt`) | 28.735 | 0.7943 | 0.2377 |
| **JDSR-NAF, perceptual phase (`best.pt`) — submitted** | **28.504** | **0.7865** | **0.1263** |

Out-of-distribution split (80 images, widened degradation parameters):

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| JDSR-NAF, fidelity phase | 27.285 | 0.7394 | 0.3270 |
| **JDSR-NAF, perceptual phase — submitted** | **27.414** | **0.7477** | **0.1813** |

The perceptual phase trades 0.23 dB PSNR for a 0.11 LPIPS improvement. Since
KLA combines all three metrics with undisclosed weights, this is favourable
under essentially any weighting; the fidelity checkpoint is shipped alongside
as an ablation.

Both checkpoints were verified from disk with `src/evaluate.py`, confirming
that EMA and raw weights had converged to within 0.003 dB of each other. Over
the full 3 200-image set, held-out images score ~0.22 dB *higher* than
trained-on images — no measurable overfitting.

**Additional selection metric.** Checkpoints were chosen on
`(PSNR/40 + SSIM + (1 − LPIPS)) / 3` rather than PSNR alone, because
PSNR-only selection reliably picks the oversmoothed checkpoint.

## 6. Runtime

| | |
|---|---|
| Hardware measured | `<GPU model>` (development); target is NVIDIA H100 |
| Batch size | 32, grouped by input shape |
| Precision | bf16 autocast (fp16 fallback), `channels_last` |
| Measurement | `time.perf_counter()` around read → transfer → inference → write |
| Throughput | 3 200 images in 38.4 s (11.8 ms/img); model init 0.63 s |
| Projected, 400 images | ≈ 5–8 s end to end including interpreter startup |

Throughput decisions:

- **`torch.compile` is deliberately not used at inference.** Inductor warmup
  costs 30–90 s, far exceeding total inference time for a few hundred small
  images. It is used during training only.
- No `lpips`, `torchvision`, `matplotlib`, or `pandas` import anywhere in
  `run.py` — `lpips` alone pulls a 233 MB weight download.
- Threaded parallel reads and a background writer thread overlap disk I/O with
  GPU compute; `cudnn.benchmark` and TF32 are enabled.

## 7. External resources

| Resource | Use | Licence | Link |
|---|---|---|---|
| PyTorch | framework | BSD-3-Clause | https://pytorch.org |
| NumPy | array I/O | BSD-3-Clause | https://numpy.org |
| `lpips` (Zhang et al., CVPR 2018) | **training loss and evaluation only** | BSD-2-Clause | https://github.com/richzhang/PerceptualSimilarity |
| torchvision AlexNet, ImageNet weights | LPIPS backbone, **training only** | BSD-3-Clause | https://pytorch.org/vision |

**No external image datasets were used.** All training data derives from the
official KLA training set, either as provided pairs or as synthetic pairs
generated from the provided ground truth. **The submitted inference path
(`run.py` + `models/best.pt`) has no pretrained-weight dependency and requires
no network access.**

Architectural components are reimplemented from published work: NAFNet (Chen
et al., ECCV 2022), Restormer MDTA/GDFN (Zamir et al., CVPR 2022), ICNR
(Aitken et al., 2017). No third-party model weights are loaded.

## 8. Reproducing

```bash
pip install -r configs/requirements-train.txt

# Recover degradation parameters from the provided pairs
python src/calibrate_degradation.py --gt_dir <GT> --noisy_dir <NoisyLR> \
    --out_json configs/calibration.json

# Phase 1 — fidelity
python train.py --gt_dir <GT> --noisy_dir <NoisyLR> \
  --preset base --phase fidelity --epochs 30 --batch_size 16 --lr_patch 96 \
  --lr 4e-4 --min_lr_frac 0.05 --warmup_steps 300 \
  --p_cutmix 0.0 --p_mixed_degradation 0.08 --p_in_dist 0.78 \
  --w_ssim 0.15 --select_on id --val_every 2 --out_dir ./runs/phase1

# Phase 1 continued at lower LR
python train.py ... --init_from ./runs/phase1/best.pt --epochs 20 --lr 1e-4 \
  --out_dir ./runs/phase1_extended

# Phase 2 — perceptual
python train.py ... --phase perceptual --init_from ./runs/phase1_extended/best.pt \
  --epochs 15 --lr 4e-5 --w_lpips 0.12 --lpips_start_frac 0.0 \
  --select_on ood --out_dir ./runs/phase2
```

Seed 42 throughout (`seed_everything`, plus per-worker seeding). `cudnn.benchmark`
is enabled for speed, so results are reproducible to within ~0.01 dB rather
than bit-exact.

## 9. Limitations and next steps

- **No 512→256 training pairs exist.** Every provided pair is 256→128.
  Content-scale jitter partially compensates, but performance on 512×512
  targets is extrapolated rather than validated. Adding an external
  high-resolution corpus degraded through the calibrated pipeline is the
  clearest remaining improvement.
- **The model is fully converged at this capacity.** Train and validation
  metrics were flat to four decimal places over the final 16 epochs with no
  train/val divergence. A larger preset (26 M) and a longer schedule are the
  obvious next experiment; both were prepared but not run within the time
  available.
- **The additive noise component is imperfectly modelled.** The measured
  variance is best fit by a signal-proportional (shot-like) term rather than a
  constant Gaussian floor, and the simulator approximates this by applying it
  at HR before downsampling. The residual autocorrelation matches
  (−0.057 measured vs −0.055 predicted), but the true generating process is
  not disclosed.
- **Failure modes.** Highest error occurs on dense high-frequency periodic
  structure, where the network cannot distinguish genuine texture from speckle
  and slightly oversmooths. See `results/examples/`.
