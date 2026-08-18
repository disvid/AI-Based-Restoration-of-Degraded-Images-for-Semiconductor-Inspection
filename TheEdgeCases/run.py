"""
run.py — KLA SEMICON 2026 submission entry point.

    python run.py <input-dir> <output-dir>

Reads every .npy file from <input-dir>, restores it, and writes one .npy of
the same filename to <output-dir> at 2x the input resolution.

Output contract (per the organizer checklist):
  * one output file per input file, identical filename
  * grayscale array of shape (H, W)
  * float32, all values within [0, 1], no NaN or Inf
  * output directory created if it does not exist

Runtime notes:
  * No torch.compile. Inductor warmup costs 30-90s, far exceeding the actual
    inference time for a few hundred small images. This is deliberate.
  * No lpips / torchvision / matplotlib / pandas imports anywhere. The
    submission must run with no internet access and no model downloads.
  * Weights are resolved relative to THIS FILE, not the working directory,
    so the script works when invoked from anywhere.
  * Threaded reads and writes overlap disk I/O with GPU compute; batching is
    grouped by shape so mixed 128x128 / 256x256 test sets work.
  * Falls back to CPU automatically when no GPU is present.
"""
import gzip
import argparse
import math
import os
import queue
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PRESETS = {
    "tiny":  dict(width=32, enc_blk_nums=(2, 2, 2), dw_kernels=(3, 5, 7),
                  bottleneck_blk_num=4, bottleneck_heads=8, dec_blk_nums=(2, 2, 2)),
    "base":  dict(width=48, enc_blk_nums=(2, 2, 4), dw_kernels=(3, 5, 7),
                  bottleneck_blk_num=6, bottleneck_heads=8, dec_blk_nums=(2, 2, 2)),
    "large": dict(width=64, enc_blk_nums=(4, 4, 6), dw_kernels=(3, 5, 7),
                  bottleneck_blk_num=8, bottleneck_heads=8, dec_blk_nums=(4, 4, 4)),
}


# ===================================================================== #
# Architecture (inference paths only)
# ===================================================================== #

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) * torch.rsqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class DropPath(nn.Module):
    """Identity at inference; present so state_dict keys line up."""
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        return x


class NAFBlock(nn.Module):
    def __init__(self, channels, dw_kernel=3, expand_ratio=2,
                 ffn_expand_ratio=2, drop_path=0.0):
        super().__init__()
        dw_channels = channels * expand_ratio
        pad = dw_kernel // 2
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(dw_channels, dw_channels, kernel_size=dw_kernel,
                                padding=pad, groups=dw_channels)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channels // 2)
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)

        ffn_channels = channels * ffn_expand_ratio
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        y = self.conv1(self.norm1(x))
        y = self.dwconv(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv2(y)
        x = x + y * self.beta
        y = self.conv3(self.norm2(x))
        y = self.sg2(y)
        y = self.conv4(y)
        return x + y * self.gamma


class MDTA(nn.Module):
    """Channel-wise attention; interior forced to fp32 for numerical safety."""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3,
                                    padding=1, groups=channels * 3)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        hd = c // self.num_heads
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = F.normalize(q.float().reshape(b, self.num_heads, hd, h * w), dim=-1)
            k = F.normalize(k.float().reshape(b, self.num_heads, hd, h * w), dim=-1)
            v = v.float().reshape(b, self.num_heads, hd, h * w)
            attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
            out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out.to(x.dtype))


class GDFN(nn.Module):
    def __init__(self, channels, expand_ratio=2.66):
        super().__init__()
        hidden = int(channels * expand_ratio)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3,
                                padding=1, groups=hidden * 2)
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        a, b = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(a) * b)


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=8, ffn_expand_ratio=2.66, drop_path=0.0):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GDFN(channels, ffn_expand_ratio)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.shuffle(self.conv(x))


class SRHead(nn.Module):
    def __init__(self, channels, out_channels=1, scale=2):
        super().__init__()
        steps = int(round(math.log2(scale)))
        layers = []
        for _ in range(steps):
            layers += [nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
                       nn.PixelShuffle(2)]
        self.up = nn.Sequential(*layers)
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = SimpleGate()
        self.out_conv = nn.Conv2d(channels // 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        x = self.act(self.refine(x))
        return self.out_conv(x)


class SymlogStem(nn.Module):
    """Raw intensity + symlog(x) = sign(x)*log1p(|x|), defined on all of R."""
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Conv2d(2, width, kernel_size=3, padding=1)

    @staticmethod
    def symlog(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def forward(self, x):
        return self.proj(torch.cat([x, self.symlog(x)], dim=1))


class JDSRNAF(nn.Module):
    def __init__(self, width=48, enc_blk_nums=(2, 2, 4), dw_kernels=(3, 5, 7),
                 bottleneck_blk_num=6, bottleneck_heads=8, dec_blk_nums=(2, 2, 2),
                 out_channels=1, sr_scale=2, residual_base="bicubic",
                 drop_path_rate=0.0, clamp_output=True):
        super().__init__()
        self.sr_scale = sr_scale
        self.residual_base = residual_base
        self.clamp_output = clamp_output
        self.pad_multiple = 2 ** len(enc_blk_nums)

        self.stem = SymlogStem(width)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        chan = width
        for num, k in zip(enc_blk_nums, dw_kernels):
            self.encoders.append(nn.Sequential(
                *[NAFBlock(chan, dw_kernel=k) for _ in range(num)]))
            self.downs.append(Downsample(chan))
            chan *= 2

        self.bottleneck = nn.Sequential(*[
            TransformerBlock(chan, num_heads=bottleneck_heads)
            for _ in range(bottleneck_blk_num)])

        self.decoders, self.ups = nn.ModuleList(), nn.ModuleList()
        for num, k in zip(dec_blk_nums, reversed(dw_kernels)):
            self.ups.append(Upsample(chan))
            chan //= 2
            self.decoders.append(nn.Sequential(
                *[NAFBlock(chan, dw_kernel=k) for _ in range(num)]))

        self.sr_head = SRHead(chan, out_channels=out_channels, scale=sr_scale)
        self.res_scale = nn.Parameter(torch.ones(1)) if residual_base == "bicubic" else None

    def _pad(self, x):
        h, w = x.shape[-2:]
        m = self.pad_multiple
        ph, pw = (m - h % m) % m, (m - w % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, (h, w)

    def forward(self, x):
        s = self.sr_scale
        # Bicubic base from the UNPADDED input: the training-set downsampler was
        # measured to be exactly torch bicubic (antialias=False), so this is the
        # closed-form pseudo-inverse and the network predicts only the residual.
        base = None
        if self.residual_base == "bicubic":
            base = F.interpolate(x.float(), scale_factor=s, mode="bicubic",
                                 align_corners=False, antialias=False)
        y, (h, w) = self._pad(x)
        y = self.stem(y)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            y = enc(y)
            skips.append(y)
            y = down(y)
        y = self.bottleneck(y)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            y = up(y)
            y = y + skip
            y = dec(y)
        y = self.sr_head(y)[..., : h * s, : w * s]
        if base is not None:
            y = y.float() + self.res_scale * base
        return y.clamp(0.0, 1.0) if self.clamp_output else y


# ===================================================================== #
# Checkpoint loading
# ===================================================================== #

def load_model(weights_path, device, prefer_ema=True,
               preset_override=None, scale_override=None, verbose=True):
    """
    Loads model weights. Prefers the EMA shadow when the checkpoint carries
    one; slimmed submission checkpoints have EMA already baked into
    model_state_dict, so both paths give the same result.
    """
    # Open the gzip file safely in read-binary mode
    with gzip.open(weights_path, 'rb') as f:
        ck = torch.load(f, map_location=device)

    preset = preset_override or ck.get("preset", "base")
    scale = scale_override or ck.get("scale", 2)
    if preset not in PRESETS:
        raise ValueError(f"unknown preset '{preset}'; expected one of {list(PRESETS)}")

    model = JDSRNAF(**PRESETS[preset], sr_scale=scale, clamp_output=True)

    sd = ck.get("model_state_dict", ck)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in sd.items()}
    missing, _ = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(
            f"checkpoint is missing {len(missing)} parameters (e.g. {missing[:4]}). "
            f"Preset mismatch? checkpoint says '{preset}'.")

    used = "weights"
    if prefer_ema and ck.get("ema_state_dict"):
        shadow = ck["ema_state_dict"]["shadow"]
        tgt = model.state_dict()
        model.load_state_dict({k: v.to(tgt[k].dtype)
                               for k, v in shadow.items() if k in tgt}, strict=False)
        used = "weights+ema"

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if verbose:
        print(f"[load] {os.path.basename(weights_path)} | preset={preset} "
              f"scale={scale}x | {used} | epoch={ck.get('epoch', '?')}",
              file=sys.stderr)
    return model


# ===================================================================== #
# I/O
# ===================================================================== #

def read_all(paths, n_workers=4):
    """Parallel .npy reads. Returns (items, errors)."""
    out_q = queue.Queue()

    def work(chunk):
        for p in chunk:
            try:
                out_q.put((os.path.basename(p),
                           np.squeeze(np.load(p)).astype(np.float32)))
            except Exception as e:
                out_q.put((os.path.basename(p), e))

    n_workers = max(1, min(n_workers, len(paths)))
    threads = [threading.Thread(target=work, args=(paths[i::n_workers],), daemon=True)
               for i in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    items, errors = [], []
    while not out_q.empty():
        name, val = out_q.get()
        (errors if isinstance(val, Exception) else items).append((name, val))
    return items, errors


def writer_thread(in_q, out_dir):
    while True:
        item = in_q.get()
        if item is None:
            break
        name, arr = item
        np.save(os.path.join(out_dir, name), arr)


def group_by_shape(items, max_batch):
    """The test set may mix 128x128 and 256x256; those can't be collated."""
    buckets = {}
    for name, arr in items:
        buckets.setdefault(arr.shape, []).append((name, arr))
    for shape, group in buckets.items():
        for i in range(0, len(group), max_batch):
            yield shape, group[i:i + max_batch]


# ===================================================================== #
# Main
# ===================================================================== #

def main():
    t_start = time.perf_counter()

    ap = argparse.ArgumentParser(
        description="KLA image restoration — python run.py <input-dir> <output-dir>")
    ap.add_argument("input_dir", help="directory containing degraded .npy files")
    ap.add_argument("output_dir", help="directory to write restored .npy files")
    ap.add_argument("--weights", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "best.pt.gz"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--preset", default=None, choices=list(PRESETS))
    ap.add_argument("--scale", type=int, default=None)
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="disable mixed precision")
    ap.add_argument("--read_workers", type=int, default=4)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"ERROR: input directory not found: {args.input_dir}")
    if not os.path.isfile(args.weights):
        sys.exit(f"ERROR: weights not found: {args.weights}")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    amp_dtype = None
    if not args.fp32 and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = load_model(args.weights, device, prefer_ema=not args.no_ema,
                       preset_override=args.preset, scale_override=args.scale,
                       verbose=not args.quiet)
    t_model = time.perf_counter()

    files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(".npy"))
    if not files:
        sys.exit(f"ERROR: no .npy files found in {args.input_dir}")
    paths = [os.path.join(args.input_dir, f) for f in files]

    items, errors = read_all(paths, args.read_workers)
    for name, err in errors:
        print(f"[warn] could not read {name}: {err}", file=sys.stderr)

    write_q = queue.Queue(maxsize=256)
    wt = threading.Thread(target=writer_thread, args=(write_q, args.output_dir),
                          daemon=True)
    wt.start()

    n_done = 0
    with torch.inference_mode():
        for _shape, group in group_by_shape(items, args.batch_size):
            batch = np.stack([a for _, a in group])[:, None]
            x = torch.from_numpy(batch).to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                y = model(x)

            # Output contract: float32, strictly in [0,1], finite everywhere.
            y = y.float()
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
            y = y.clamp(0.0, 1.0).cpu().numpy()[:, 0].astype(np.float32)

            for (name, _), out in zip(group, y):
                write_q.put((name, out))
            n_done += len(group)

    write_q.put(None)
    wt.join()

    t_end = time.perf_counter()
    if not args.quiet:
        print(f"[done] {n_done} images | init {t_model - t_start:.2f}s | "
              f"total {t_end - t_start:.2f}s | "
              f"{1000 * (t_end - t_model) / max(n_done, 1):.1f} ms/img "
              f"| device={device.type} amp={amp_dtype}", file=sys.stderr)

    if n_done != len(files):
        print(f"[warn] wrote {n_done} outputs for {len(files)} inputs", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
