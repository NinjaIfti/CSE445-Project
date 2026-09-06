"""
dasfnet.py
===========================================================================
DASF-Net -- Dual-Adversarial Spectral-Fusion Network.

Reference implementation of the core methodology described in main.tex /
main.pdf.  This file implements the architecture, the objective and the
three-stage training algorithm; it is the executable counterpart of Fig. 1,
Fig. 2, Fig. 3 and Algorithms 1-2 of the paper.

The premise: a deepfake detector fails across languages and fails under
adversarial perturbation for the *same* reason -- it has latched onto a
nuisance direction in feature space.  Both are therefore attacked with an
adversary, on one shared embedding:

    L_total = L_det(x)                     detection
            + lam_adv * L_det(x + delta)   perturbation adversary  (local)
            + lam_syn * L_syn              synthesis-family aux head
            - lam_d   * L_lang             domain adversary, via GRL (global)

Pipeline (Fig. 1):

    audio (22.05 kHz retained)
      |-- Branch-S : STFT -> linear FB(20) -> DCT -> 20 coeff -> d + dd
      |-- Branch-D : STFT -> mel FB(128)   -> log-power       -> d + dd
              both -> 299x299x3 -> adapter stem -> SHARED Xception trunk
              -> CASF (two-way cross-attention + learned gate) -> z in R^256
              -> h_det / h_syn / GRL->h_lang

Three-stage curriculum (Fig. 3):

    Stage 1  representation warm-up      trunk frozen, L_det only
    Stage 2  cross-lingual alignment     trunk unfrozen, + h_syn, + GRL h_lang
    Stage 3  dual-adversarial hardening  + adaptive FGSM/PGD sampling

Dependencies:
    pip install torch torchvision timm librosa soundfile scikit-learn \
                numpy pandas tqdm

Quick check with no data and no downloads:
    python dasfnet.py --smoke-test

Real runs:
    python dasfnet.py --stage all \
        --source_manifest wavefake_manifest.csv \
        --target_manifest banglafake_manifest.csv

Manifest CSV (same format as deepfake_audio_pipeline.py):
    filepath,label,vocoder,language
    /data/wavefake/ljspeech_melgan/LJ001-0001.wav,1,melgan,en
    /data/banglafake/real/0002.wav,0,real,bn

    label    : 0 = bonafide/real, 1 = deepfake
    vocoder  : synthesis method ("real" for genuine audio)
    language : "en" (source) or "bn" (target)

NOTE ON UNRUN CODE: none of this has been trained -- there are no results in
the paper and none here.  The smoke test exercises every code path end to end
on random tensors, which verifies shapes and gradient flow, not accuracy.
===========================================================================
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from sklearn.metrics import roc_curve


# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

@dataclass
class AudioConfig:
    """High-Band-Preserving (HBP) preprocessing -- Section IV-A of the paper.

    The native 22.05 kHz rate is RETAINED.  Downsampling to 16 kHz (as in the
    prior single-branch work) imposes an 8 kHz Nyquist ceiling and deletes the
    8-11 kHz band, which is where vocoder and end-to-end TTS artifacts are most
    pronounced and where the two generator families most differ.
    """
    sample_rate: int = 22050          # NOT 16000 -- this is the point
    target_duration_s: float = 4.0
    silence_gap_s: float = 2.0
    silence_db_threshold: float = 40.0

    # STFT is shared by both branches, so any difference between them is
    # attributable to the filterbank and compression stages alone.
    n_fft: int = 2048
    hop_length: int = 512

    # Branch-S -- sparse cepstral
    lfcc_n_filters: int = 20
    lfcc_n_coeff: int = 20
    lfcc_fmin: float = 0.0
    lfcc_fmax: float = 11025.0        # = sample_rate / 2

    # Branch-D -- dense spectral
    mel_n_mels: int = 128
    mel_fmax: float = 11025.0

    cnn_input_size: int = 299         # Xception input geometry


@dataclass
class ModelConfig:
    backbone: str = "xception"        # timm name; ImageNet-pretrained
    trunk_channels: int = 2048        # Xception exit-flow width
    n_tokens: int = 100               # 10x10 spatial grid at 299x299 input
    d_model: int = 256                # CASF token width
    n_heads: int = 4
    n_synthesis_families: int = 3     # {real, GAN-vocoder, E2E-TTS}
    n_languages: int = 2              # {en, bn}
    head_hidden: int = 128
    adapter_channels: int = 32


@dataclass
class TrainConfig:
    # The shared trunk is evaluated twice per utterance, so one sample costs
    # roughly double the activation memory of a single-branch Xception at
    # 299x299.  A batch of 128 does NOT fit in 8 GB (RTX 5060 Ti).  The
    # micro-batch below fits; `accum_steps` restores an effective batch of 128.
    #
    # WARNING: gradient accumulation is declared here but is NOT yet wired
    # into the three training loops -- they step the optimiser every batch.
    # Until it is implemented, training runs at an effective batch of 16, which
    # does not match the learning rates quoted in the report.
    batch_size: int = 16              # micro-batch that fits in 8 GB
    accum_steps: int = 8              # 16 x 8 = 128 effective
    label_smoothing: float = 0.1

    stage1_epochs: int = 5            # representation warm-up
    stage2_epochs: int = 15           # cross-lingual alignment
    stage3_epochs: int = 10           # dual-adversarial hardening

    lr_stage1: float = 1e-4
    lr_stage2: float = 1e-4
    lr_stage3: float = 5e-5           # reduced for hardening

    lam_syn: float = 0.3              # synthesis-family aux weight
    lam_d_max: float = 1.0            # GRL ceiling
    lam_adv: float = 1.0              # adversarial detection term
    grl_gamma: float = 10.0           # DANN ramp steepness

    mixed_precision: bool = True
    num_workers: int = 4
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class AttackConfig:
    """Unified threat model -- Section IV-G of the paper.

    Both attacks are swept over the SAME epsilon grid, in the normalised
    feature domain.  The prior work reported FGSM at eps ~1e-3 and PGD at
    eps ~1e-1 -- budgets two orders of magnitude apart, which makes the two
    attacks non-comparable.  One grid, reported per epsilon.
    """
    epsilons: Tuple[float, ...] = (0.001, 0.005, 0.010)
    pgd_steps: int = 10
    pgd_alpha_ratio: float = 0.25     # alpha = eps / 4
    momentum: float = 0.2             # m, attack-sampling momentum
    clean_fraction: float = 1.0 / 3.0 # p_c, clean share of every minibatch


AUDIO_CFG = AudioConfig()
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()
ATTACK_CFG = AttackConfig()

# Synthesis-family label map.  Deliberately COARSE: predicting the specific
# generator invites memorising generator identity, which is exactly the
# failure that breaks cross-vocoder transfer.  Family pushes z to organise
# around mechanism instead.
FAMILY_REAL, FAMILY_GAN, FAMILY_E2E = 0, 1, 2

_GAN_VOCODERS = {
    "melgan", "melgan_large", "melgan-l", "parallel_wavegan", "pwg",
    "multi_band_melgan", "mb-melgan", "mb_melgan", "full_band_melgan",
    "fb-melgan", "fb_melgan", "hifigan", "hifi-gan", "waveglow",
}
_E2E_SYSTEMS = {"vits", "tts", "fastspeech2", "tacotron2"}


def synthesis_family(vocoder: str) -> int:
    v = (vocoder or "").strip().lower().replace(" ", "_")
    if v in ("real", "bonafide", "genuine", ""):
        return FAMILY_REAL
    if v in _GAN_VOCODERS:
        return FAMILY_GAN
    if v in _E2E_SYSTEMS:
        return FAMILY_E2E
    # Unknown generator: treat as end-to-end rather than silently calling it
    # real, and make the assumption visible.
    print(f"[warn] unmapped vocoder '{vocoder}' -> FAMILY_E2E")
    return FAMILY_E2E


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# 2. PREPROCESSING AND DUAL-VIEW FEATURE EXTRACTION  (Fig. 1, rows 2-3)
# ===========================================================================

def load_audio_hbp(filepath: str, cfg: AudioConfig = AUDIO_CFG) -> np.ndarray:
    """Load mono at the NATIVE rate, trim long silences, fix duration to 4 s."""
    if librosa is None:
        raise ImportError("librosa is required for audio loading")

    y, _ = librosa.load(filepath, sr=cfg.sample_rate, mono=True)

    # Trim leading/trailing silence, then drop interior gaps longer than
    # silence_gap_s by splitting on non-silent intervals.
    intervals = librosa.effects.split(y, top_db=cfg.silence_db_threshold)
    if len(intervals) > 0:
        max_gap = int(cfg.silence_gap_s * cfg.sample_rate)
        kept, prev_end = [], None
        for start, end in intervals:
            if prev_end is not None and (start - prev_end) < max_gap:
                kept.append(y[prev_end:start])       # short gap: keep it
            kept.append(y[start:end])
            prev_end = end
        y = np.concatenate(kept) if kept else y

    target_len = int(cfg.target_duration_s * cfg.sample_rate)
    if len(y) >= target_len:                          # centre crop
        offset = (len(y) - target_len) // 2
        y = y[offset:offset + target_len]
    else:                                             # reflection pad
        pad = target_len - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2), mode="reflect")
    return y.astype(np.float32)


def _add_deltas(feat: np.ndarray) -> np.ndarray:
    """Stack [static; delta; delta-delta] along the first axis."""
    d1 = librosa.feature.delta(feat, order=1)
    d2 = librosa.feature.delta(feat, order=2)
    return np.concatenate([feat, d1, d2], axis=0)


def _to_cnn_input(feat: np.ndarray, size: int = AUDIO_CFG.cnn_input_size) -> np.ndarray:
    """(C, F, T) or (F, T) -> (3, size, size), per-utterance standardised.

    Standardisation matters: the adversarial threat model of Section IV-G is
    defined in this normalised domain, so epsilon means the same thing for
    every utterance and for both branches.
    """
    t = torch.from_numpy(np.asarray(feat, dtype=np.float32))
    if t.dim() == 2:
        t = t.unsqueeze(0)
    t = F.interpolate(t.unsqueeze(0), size=(size, size),
                      mode="bilinear", align_corners=False).squeeze(0)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    elif t.shape[0] != 3:
        t = t[:3] if t.shape[0] > 3 else t.repeat(3 // t.shape[0] + 1, 1, 1)[:3]

    t = (t - t.mean()) / (t.std() + 1e-8)
    t = torch.sigmoid(t)               # squash into [0, 1]
    return t.numpy()


def extract_lfcc(y: np.ndarray, cfg: AudioConfig = AUDIO_CFG) -> np.ndarray:
    """Branch-S -- sparse cepstral.  Linear FB -> DCT -> 20 coeff -> d + dd."""
    spec = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length)) ** 2

    # Linear-frequency triangular filterbank (contrast: mel is log-spaced).
    freqs = librosa.fft_frequencies(sr=cfg.sample_rate, n_fft=cfg.n_fft)
    edges = np.linspace(cfg.lfcc_fmin, cfg.lfcc_fmax, cfg.lfcc_n_filters + 2)
    fb = np.zeros((cfg.lfcc_n_filters, len(freqs)), dtype=np.float32)
    for i in range(cfg.lfcc_n_filters):
        lo, ctr, hi = edges[i], edges[i + 1], edges[i + 2]
        rising = (freqs >= lo) & (freqs <= ctr)
        falling = (freqs > ctr) & (freqs <= hi)
        fb[i, rising] = (freqs[rising] - lo) / max(ctr - lo, 1e-8)
        fb[i, falling] = (hi - freqs[falling]) / max(hi - ctr, 1e-8)

    log_fb = np.log(fb @ spec + 1e-10)
    from scipy.fftpack import dct as _dct
    lfcc = _dct(log_fb, type=2, axis=0, norm="ortho")[:cfg.lfcc_n_coeff]
    return _add_deltas(lfcc)                          # (60, T)


def extract_logmel(y: np.ndarray, cfg: AudioConfig = AUDIO_CFG) -> np.ndarray:
    """Branch-D -- dense spectral.  Mel FB -> log-power, NO DCT -> d + dd.

    Skipping the DCT is the whole point: it preserves the dense 2-D
    time-frequency structure that convolution and ImageNet transfer assume.
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, n_mels=cfg.mel_n_mels, fmax=cfg.mel_fmax,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return _add_deltas(log_mel)                       # (384, T) = 3 x 128


class DualViewDataset(Dataset):
    """Yields BOTH spectral views of one utterance, plus all three labels."""

    def __init__(self, manifest_path: str, cfg: AudioConfig = AUDIO_CFG):
        if pd is None:
            raise ImportError("pandas is required to read manifests")
        self.df = pd.read_csv(manifest_path)
        for col in ("filepath", "label", "vocoder", "language"):
            if col not in self.df.columns:
                raise ValueError(f"manifest missing required column '{col}'")
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y = load_audio_hbp(row["filepath"], self.cfg)

        x_s = _to_cnn_input(extract_lfcc(y, self.cfg), self.cfg.cnn_input_size)
        x_d = _to_cnn_input(extract_logmel(y, self.cfg), self.cfg.cnn_input_size)

        return (
            torch.from_numpy(x_s),
            torch.from_numpy(x_d),
            torch.tensor(int(row["label"]), dtype=torch.long),
            torch.tensor(synthesis_family(row["vocoder"]), dtype=torch.long),
            torch.tensor(0 if str(row["language"]).lower() == "en" else 1,
                         dtype=torch.long),
        )


# ===========================================================================
# 3. MODEL  (Fig. 2)
# ===========================================================================

class _GradientReversal(torch.autograd.Function):
    """Identity forward, negated-and-scaled gradient backward."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradientReversal.apply(x, lambd)


def grl_lambda(progress: float, lam_max: float, gamma: float) -> float:
    """DANN ramp: lam_d(p) = lam_max * (2 / (1 + exp(-gamma p)) - 1).

    Ramping matters.  A gradient-reversed discriminator attached to an
    untrained encoder produces a large, uninformative reversed gradient, so
    the adversary is given influence only as it becomes informative.
    """
    p = float(np.clip(progress, 0.0, 1.0))
    return lam_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


class AdapterStem(nn.Module):
    """Per-branch entry adapter: maps branch-specific input statistics into a
    common range before the SHARED trunk sees them.  ~0.06 M parameters."""

    def __init__(self, ch: int = MODEL_CFG.adapter_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 3, 3, padding=1, bias=False),
            nn.BatchNorm2d(3), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x       # residual: start near the identity


class CASF(nn.Module):
    """Cross-Attention Spectral Fusion.

    Each view queries the other, then a LEARNED GATE mixes them:

        g = sigmoid(W_g [h_D ; h_S])
        z = pool( g * h_D + (1 - g) * h_S )

    A fixed concatenation would assert that both views always matter equally.
    The gate instead lets the network decide, per utterance and per dimension,
    which representation carries the artifact -- and the distribution of g is
    itself a reportable result (experiment E5).
    """

    def __init__(self, cfg: ModelConfig = MODEL_CFG):
        super().__init__()
        self.proj = nn.Linear(cfg.trunk_channels, cfg.d_model)
        self.attn_d = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.attn_s = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.norm_d = nn.LayerNorm(cfg.d_model)
        self.norm_s = nn.LayerNorm(cfg.d_model)
        self.gate = nn.Linear(2 * cfg.d_model, cfg.d_model)

    @staticmethod
    def _tokens(f: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, H*W, C)."""
        b, c, h, w = f.shape
        return f.flatten(2).transpose(1, 2)

    def forward(self, f_s: torch.Tensor, f_d: torch.Tensor):
        t_s = self.proj(self._tokens(f_s))            # (B, N, d)
        t_d = self.proj(self._tokens(f_d))

        h_d, _ = self.attn_d(t_d, t_s, t_s)           # Mel queries LFCC
        h_d = self.norm_d(t_d + h_d)
        h_s, _ = self.attn_s(t_s, t_d, t_d)           # LFCC queries Mel
        h_s = self.norm_s(t_s + h_s)

        g = torch.sigmoid(self.gate(torch.cat([h_d, h_s], dim=-1)))
        z = g * h_d + (1.0 - g) * h_s
        return z.mean(dim=1), g                       # (B, d), (B, N, d)


def _build_trunk(cfg: ModelConfig = MODEL_CFG) -> nn.Module:
    """ImageNet-pretrained Xception with the classifier and pooling removed,
    so it emits a (B, 2048, 10, 10) feature map at 299x299 input."""
    if timm is None:
        raise ImportError("timm is required for the Xception backbone")
    for name in (cfg.backbone, "legacy_xception"):
        try:
            return timm.create_model(name, pretrained=True,
                                     num_classes=0, global_pool="")
        except Exception:
            continue
    raise RuntimeError(f"could not construct backbone '{cfg.backbone}'")


def _mlp_head(d_in: int, d_hidden: int, d_out: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.ReLU(inplace=True), nn.Linear(d_hidden, d_out)
    )


class DASFNet(nn.Module):
    """Dual-branch encoder + CASF + three heads.  Algorithm 1 of the paper."""

    def __init__(self, cfg: ModelConfig = MODEL_CFG, trunk: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.adapter_s = AdapterStem(cfg.adapter_channels)
        self.adapter_d = AdapterStem(cfg.adapter_channels)

        # ONE trunk, applied to both branches with tied weights.  Two
        # independent backbones would let each branch build a private feature
        # space, and CASF would then have to reconcile two unrelated
        # geometries.  Sharing forces a single artifact space, for 0.12 M
        # extra adapter parameters instead of a second 23 M model.
        self.trunk = trunk if trunk is not None else _build_trunk(cfg)

        self.casf = CASF(cfg)
        self.h_det = _mlp_head(cfg.d_model, cfg.head_hidden, 2)
        self.h_syn = _mlp_head(cfg.d_model, cfg.head_hidden, cfg.n_synthesis_families)
        self.h_lang = _mlp_head(cfg.d_model, cfg.head_hidden, cfg.n_languages)

    def freeze_trunk(self) -> None:
        for p in self.trunk.parameters():
            p.requires_grad = False

    def unfreeze_trunk(self) -> None:
        for p in self.trunk.parameters():
            p.requires_grad = True

    def embed(self, x_s: torch.Tensor, x_d: torch.Tensor):
        f_s = self.trunk(self.adapter_s(x_s))
        f_d = self.trunk(self.adapter_d(x_d))         # same weights
        return self.casf(f_s, f_d)

    def forward(self, x_s: torch.Tensor, x_d: torch.Tensor, lam_d: float = 0.0):
        z, gate = self.embed(x_s, x_d)
        return {
            "det": self.h_det(z),
            "syn": self.h_syn(z),
            "lang": self.h_lang(grad_reverse(z, lam_d)),
            "z": z,
            "gate": gate,
        }


# ===========================================================================
# 4. ADVERSARIAL ATTACKS AND ADAPTIVE SAMPLING  (Sections IV-G, V-B)
# ===========================================================================

def _det_loss(model: DASFNet, x_s, x_d, y, smoothing: float = 0.0) -> torch.Tensor:
    return F.cross_entropy(model(x_s, x_d, lam_d=0.0)["det"], y,
                           label_smoothing=smoothing)


def fgsm_attack(model: DASFNet, x_s, x_d, y, eps: float):
    """Single-step attack applied to BOTH views simultaneously."""
    x_s = x_s.clone().detach().requires_grad_(True)
    x_d = x_d.clone().detach().requires_grad_(True)
    loss = _det_loss(model, x_s, x_d, y)
    g_s, g_d = torch.autograd.grad(loss, [x_s, x_d])
    return (
        (x_s + eps * g_s.sign()).clamp(0, 1).detach(),
        (x_d + eps * g_d.sign()).clamp(0, 1).detach(),
    )


def pgd_attack(model: DASFNet, x_s, x_d, y, eps: float,
               steps: int = ATTACK_CFG.pgd_steps,
               alpha_ratio: float = ATTACK_CFG.pgd_alpha_ratio):
    """Iterative attack, projected onto the l_inf ball after each step."""
    alpha = eps * alpha_ratio
    x0_s, x0_d = x_s.clone().detach(), x_d.clone().detach()
    a_s = (x0_s + torch.empty_like(x0_s).uniform_(-eps, eps)).clamp(0, 1)
    a_d = (x0_d + torch.empty_like(x0_d).uniform_(-eps, eps)).clamp(0, 1)

    for _ in range(steps):
        a_s.requires_grad_(True)
        a_d.requires_grad_(True)
        loss = _det_loss(model, a_s, a_d, y)
        g_s, g_d = torch.autograd.grad(loss, [a_s, a_d])
        a_s = (a_s.detach() + alpha * g_s.sign())
        a_d = (a_d.detach() + alpha * g_d.sign())
        a_s = (x0_s + (a_s - x0_s).clamp(-eps, eps)).clamp(0, 1)
        a_d = (x0_d + (a_d - x0_d).clamp(-eps, eps)).clamp(0, 1)

    return a_s.detach(), a_d.detach()


class AdaptiveAttackSampler:
    """Momentum-smoothed attack selection -- Section V-B.

        s_a <- (1 - m) s_a + m * s_hat_a

    Attacks are drawn in proportion to s_a, so training concentrates on
    whichever attack currently works, without a hand-tuned schedule.
    """

    def __init__(self, cfg: AttackConfig = ATTACK_CFG):
        self.cfg = cfg
        self.success: Dict[str, float] = {"fgsm": 0.5, "pgd": 0.5}

    def sample_attack(self) -> str:
        names = list(self.success)
        weights = np.array([max(self.success[n], 1e-3) for n in names])
        return str(np.random.choice(names, p=weights / weights.sum()))

    def sample_epsilon(self) -> float:
        return float(np.random.choice(self.cfg.epsilons))

    def update(self, name: str, success_rate: float) -> None:
        m = self.cfg.momentum
        self.success[name] = (1 - m) * self.success[name] + m * success_rate

    def generate(self, model: DASFNet, x_s, x_d, y):
        """Return (adv_s, adv_d, attack_name) and refresh the success estimate."""
        name, eps = self.sample_attack(), self.sample_epsilon()
        was_training = model.training
        model.eval()                       # attacks use a deterministic model
        fn = fgsm_attack if name == "fgsm" else pgd_attack
        adv_s, adv_d = fn(model, x_s, x_d, y, eps)

        with torch.no_grad():
            pred = model(adv_s, adv_d, lam_d=0.0)["det"].argmax(1)
            self.update(name, (pred != y).float().mean().item())

        if was_training:
            model.train()
        return adv_s, adv_d, name


# ===========================================================================
# 5. METRICS  (Section VI-B)
# ===========================================================================

def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Equal Error Rate -- threshold-free, so it survives a prior mismatch
    between the 8:1 source corpus and the 1:1 target corpus."""
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    accs = []
    for c in (0, 1):
        mask = y_true == c
        if mask.sum() > 0:
            accs.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(accs)) if accs else float("nan")


# NOTE: the paper also lists min-tDCF.  t-DCF is defined over the scores of a
# *tandem* system -- a spoofing countermeasure plus an automatic speaker
# verification subsystem -- and neither WaveFake nor BanglaFake ships ASV
# scores.  It is therefore not computable from this data without adding an ASV
# system, and is deliberately not faked here.  See the note in README.md.


@torch.no_grad()
def evaluate(model: DASFNet, loader: DataLoader, device: str,
             attack: Optional[str] = None, eps: float = 0.0,
             sampler: Optional[AdaptiveAttackSampler] = None) -> Dict[str, float]:
    """Clean or under-attack evaluation.  Returns EER and balanced accuracy."""
    model.eval()
    scores, labels, preds, gates = [], [], [], []

    for x_s, x_d, y, _, _ in loader:
        x_s, x_d, y = x_s.to(device), x_d.to(device), y.to(device)

        if attack is not None:
            with torch.enable_grad():   # attacks need gradients
                fn = fgsm_attack if attack == "fgsm" else pgd_attack
                x_s, x_d = fn(model, x_s, x_d, y, eps)

        out = model(x_s, x_d, lam_d=0.0)
        prob = F.softmax(out["det"], dim=1)[:, 1]
        scores.append(prob.cpu().numpy())
        preds.append(out["det"].argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
        gates.append(out["gate"].mean(dim=(1, 2)).cpu().numpy())

    y_true = np.concatenate(labels)
    return {
        "eer": compute_eer(y_true, np.concatenate(scores)),
        "balanced_acc": balanced_accuracy(y_true, np.concatenate(preds)),
        "mean_gate": float(np.concatenate(gates).mean()),
    }


@torch.no_grad()
def gate_statistics(model: DASFNet, loader: DataLoader, device: str) -> Dict[int, Dict[str, float]]:
    """Experiment E5.  The gate distribution per synthesis family answers,
    quantitatively: do GAN vocoders and end-to-end TTS leave their traces in
    different spectral representations?  g -> 1 means the model leaned on the
    dense Mel view; g -> 0 means it leaned on the sparse cepstral view."""
    model.eval()
    per_family: Dict[int, List[float]] = {}

    for x_s, x_d, _, fam, _ in loader:
        out = model(x_s.to(device), x_d.to(device), lam_d=0.0)
        g = out["gate"].mean(dim=(1, 2)).cpu().numpy()
        for value, f in zip(g, fam.numpy()):
            per_family.setdefault(int(f), []).append(float(value))

    return {f: {"mean_gate": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
            for f, v in per_family.items()}


# ===========================================================================
# 6. THREE-STAGE TRAINING  (Algorithm 2, Fig. 3)
# ===========================================================================

def _trainable(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def stage1_warmup(model: DASFNet, src_loader: DataLoader,
                  cfg: TrainConfig = TRAIN_CFG) -> DASFNet:
    """Trunk frozen; train adapters + CASF + h_det on labelled source only.

    This buys a stable fused representation before the pretrained trunk is
    disturbed -- and before either adversary is switched on.
    """
    print("\n=== Stage 1: representation warm-up ===")
    model.to(cfg.device)
    model.freeze_trunk()
    opt = torch.optim.Adam(_trainable(model), lr=cfg.lr_stage1)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision and
                                       cfg.device.startswith("cuda"))

    for epoch in range(cfg.stage1_epochs):
        model.train()
        running = 0.0
        for x_s, x_d, y, _, _ in src_loader:
            x_s, x_d, y = x_s.to(cfg.device), x_d.to(cfg.device), y.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(x_s, x_d, lam_d=0.0)
                loss = F.cross_entropy(out["det"], y,
                                       label_smoothing=cfg.label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        print(f"  epoch {epoch + 1}/{cfg.stage1_epochs}  "
              f"L_det = {running / max(len(src_loader), 1):.4f}")
    return model


def stage2_align(model: DASFNet, src_loader: DataLoader, tgt_loader: DataLoader,
                 cfg: TrainConfig = TRAIN_CFG) -> DASFNet:
    """Unfreeze the trunk; attach h_syn and the gradient-reversed h_lang.

    Target audio enters UNLABELLED -- only its language tag is used, which
    keeps the setting honestly unsupervised with respect to the target domain.
    L_det and L_syn are computed on source samples only.
    """
    print("\n=== Stage 2: cross-lingual alignment ===")
    model.to(cfg.device)
    model.unfreeze_trunk()
    opt = torch.optim.Adam(_trainable(model), lr=cfg.lr_stage2)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision and
                                       cfg.device.startswith("cuda"))

    steps_per_epoch = min(len(src_loader), len(tgt_loader))
    total_steps = max(steps_per_epoch * cfg.stage2_epochs, 1)
    step = 0

    for epoch in range(cfg.stage2_epochs):
        model.train()
        agg = {"det": 0.0, "syn": 0.0, "lang": 0.0}

        for (xs_s, xd_s, y_s, f_s, l_s), (xs_t, xd_t, _, _, l_t) in zip(src_loader, tgt_loader):
            lam_d = grl_lambda(step / total_steps, cfg.lam_d_max, cfg.grl_gamma)

            x_s = torch.cat([xs_s, xs_t]).to(cfg.device)
            x_d = torch.cat([xd_s, xd_t]).to(cfg.device)
            y_lang = torch.cat([l_s, l_t]).to(cfg.device)
            n_src = xs_s.size(0)
            y_det = y_s.to(cfg.device)
            y_fam = f_s.to(cfg.device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(x_s, x_d, lam_d=lam_d)
                l_det = F.cross_entropy(out["det"][:n_src], y_det,
                                        label_smoothing=cfg.label_smoothing)
                l_syn = F.cross_entropy(out["syn"][:n_src], y_fam)
                l_lng = F.cross_entropy(out["lang"], y_lang)  # GRL flips its sign
                loss = l_det + cfg.lam_syn * l_syn + l_lng

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            agg["det"] += l_det.item(); agg["syn"] += l_syn.item(); agg["lang"] += l_lng.item()
            step += 1

        n = max(steps_per_epoch, 1)
        print(f"  epoch {epoch + 1}/{cfg.stage2_epochs}  lam_d = {lam_d:.3f}  "
              f"L_det = {agg['det'] / n:.4f}  L_syn = {agg['syn'] / n:.4f}  "
              f"L_lang = {agg['lang'] / n:.4f}")
    return model


def stage3_harden(model: DASFNet, src_loader: DataLoader, tgt_loader: DataLoader,
                  cfg: TrainConfig = TRAIN_CFG,
                  atk: AttackConfig = ATTACK_CFG) -> Tuple[DASFNet, AdaptiveAttackSampler]:
    """Add adaptive FGSM/PGD sampling, with the language adversary still on.

    Hardening AFTER alignment, rather than before, means the perturbation
    adversary hardens a representation that is already language-invariant.
    """
    print("\n=== Stage 3: dual-adversarial hardening ===")
    model.to(cfg.device)
    model.unfreeze_trunk()
    opt = torch.optim.Adam(_trainable(model), lr=cfg.lr_stage3)
    sampler = AdaptiveAttackSampler(atk)

    steps_per_epoch = min(len(src_loader), len(tgt_loader))

    for epoch in range(cfg.stage3_epochs):
        model.train()
        agg, seen = 0.0, 0

        for (xs_s, xd_s, y_s, f_s, l_s), (xs_t, xd_t, _, _, l_t) in zip(src_loader, tgt_loader):
            xs_s, xd_s = xs_s.to(cfg.device), xd_s.to(cfg.device)
            xs_t, xd_t = xs_t.to(cfg.device), xd_t.to(cfg.device)
            y_det = y_s.to(cfg.device)
            y_fam = f_s.to(cfg.device)
            y_lang = torch.cat([l_s, l_t]).to(cfg.device)

            # Split the source batch: p_c stays clean to preserve baseline
            # accuracy, the rest is attacked.
            n_src = xs_s.size(0)
            n_clean = max(int(round(atk.clean_fraction * n_src)), 1)
            clean_s, clean_d, y_clean = xs_s[:n_clean], xd_s[:n_clean], y_det[:n_clean]
            atk_s, atk_d, y_atk = xs_s[n_clean:], xd_s[n_clean:], y_det[n_clean:]

            if atk_s.size(0) > 0:
                adv_s, adv_d, _ = sampler.generate(model, atk_s, atk_d, y_atk)
            else:
                adv_s, adv_d = atk_s, atk_d

            lam_d = cfg.lam_d_max            # already ramped in during Stage 2

            opt.zero_grad(set_to_none=True)

            out_clean = model(torch.cat([clean_s, xs_t]),
                              torch.cat([clean_d, xd_t]), lam_d=lam_d)
            l_det = F.cross_entropy(out_clean["det"][:n_clean], y_clean,
                                    label_smoothing=cfg.label_smoothing)
            l_syn = F.cross_entropy(out_clean["syn"][:n_clean], y_fam[:n_clean])
            l_lng = F.cross_entropy(
                out_clean["lang"],
                torch.cat([y_lang[:n_clean], y_lang[n_src:]]),
            )

            if adv_s.size(0) > 0:
                l_adv = F.cross_entropy(
                    model(adv_s, adv_d, lam_d=0.0)["det"], y_atk,
                    label_smoothing=cfg.label_smoothing)
            else:
                l_adv = torch.zeros((), device=cfg.device)

            # Equation (6) of the paper.  The minus on L_lang is realised by
            # the GRL inside the model, not by subtracting here.
            loss = l_det + cfg.lam_adv * l_adv + cfg.lam_syn * l_syn + l_lng
            loss.backward()
            opt.step()

            agg += loss.item(); seen += 1

        print(f"  epoch {epoch + 1}/{cfg.stage3_epochs}  "
              f"L_total = {agg / max(seen, 1):.4f}  "
              f"attack success: FGSM {sampler.success['fgsm']:.3f} / "
              f"PGD {sampler.success['pgd']:.3f}")

    return model, sampler


# ===========================================================================
# 7. SMOKE TEST -- exercises every path on random tensors, no data, no download
# ===========================================================================

class _RandomDualViewDataset(Dataset):
    def __init__(self, n: int, language: int, size: int = 299, seed: int = 0):
        self.n, self.language, self.size = n, language, size
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        y = int(self.rng.randint(0, 2))
        fam = FAMILY_REAL if y == 0 else (FAMILY_GAN if self.language == 0 else FAMILY_E2E)
        return (
            torch.rand(3, self.size, self.size),
            torch.rand(3, self.size, self.size),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(fam, dtype=torch.long),
            torch.tensor(self.language, dtype=torch.long),
        )


class _TinyTrunk(nn.Module):
    """Stand-in for Xception so the smoke test needs no pretrained download.
    Emits (B, 2048, 10, 10), the same shape the real trunk produces."""

    def __init__(self, out_ch: int = MODEL_CFG.trunk_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=4, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 3, stride=4, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((10, 10)),
        )

    def forward(self, x):
        return self.net(x)


def smoke_test() -> None:
    print("=" * 72)
    print("DASF-Net smoke test -- random tensors, tiny trunk, 1 epoch per stage")
    print("=" * 72)

    set_seed(0)
    cfg = TrainConfig(batch_size=4, stage1_epochs=1, stage2_epochs=1,
                      stage3_epochs=1, mixed_precision=False, num_workers=0,
                      device="cpu")
    atk = AttackConfig(pgd_steps=2)

    size = 64                                     # smaller than 299, for speed
    src = DataLoader(_RandomDualViewDataset(16, language=0, size=size, seed=1),
                     batch_size=cfg.batch_size)
    tgt = DataLoader(_RandomDualViewDataset(16, language=1, size=size, seed=2),
                     batch_size=cfg.batch_size)

    model = DASFNet(MODEL_CFG, trunk=_TinyTrunk())
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nparameters (tiny trunk, not Xception): {n_params / 1e6:.2f} M")

    x_s, x_d, y, fam, lang = next(iter(src))
    out = model(x_s, x_d, lam_d=0.5)
    print(f"forward: det {tuple(out['det'].shape)}  syn {tuple(out['syn'].shape)}  "
          f"lang {tuple(out['lang'].shape)}  z {tuple(out['z'].shape)}  "
          f"gate {tuple(out['gate'].shape)}")

    # Gradient reversal must actually flip the sign reaching the trunk.
    model.zero_grad()
    F.cross_entropy(model(x_s, x_d, lam_d=1.0)["lang"], lang).backward()
    g_pos = model.adapter_d.block[0].weight.grad.clone()
    model.zero_grad()
    F.cross_entropy(model(x_s, x_d, lam_d=-1.0)["lang"], lang).backward()
    g_neg = model.adapter_d.block[0].weight.grad.clone()
    flipped = torch.allclose(g_pos, -g_neg, atol=1e-5)
    print(f"GRL sign flip verified: {flipped}")

    adv_s, adv_d = fgsm_attack(model, x_s, x_d, y, eps=0.01)
    linf = (adv_s - x_s).abs().max().item()
    print(f"FGSM l_inf = {linf:.4f} (budget 0.0100), range ok: "
          f"{bool(adv_s.min() >= 0 and adv_s.max() <= 1)}")

    adv_s, adv_d = pgd_attack(model, x_s, x_d, y, eps=0.01, steps=atk.pgd_steps)
    print(f"PGD  l_inf = {(adv_s - x_s).abs().max().item():.4f} (budget 0.0100)")

    print(f"GRL ramp: lam_d(0) = {grl_lambda(0, 1.0, 10):.3f}  "
          f"lam_d(0.5) = {grl_lambda(0.5, 1.0, 10):.3f}  "
          f"lam_d(1) = {grl_lambda(1, 1.0, 10):.3f}")

    model = stage1_warmup(model, src, cfg)
    model = stage2_align(model, src, tgt, cfg)
    model, sampler = stage3_harden(model, src, tgt, cfg, atk)

    print("\n=== evaluation ===")
    print(f"clean  : {evaluate(model, tgt, cfg.device)}")
    print(f"FGSM   : {evaluate(model, tgt, cfg.device, attack='fgsm', eps=0.005)}")
    print(f"gate by family: {gate_statistics(model, tgt, cfg.device)}")

    print("\nAll paths executed. Shapes and gradient flow verified; accuracy is")
    print("meaningless here -- the inputs are random noise.")


# ===========================================================================
# 8. ENTRY POINT
# ===========================================================================

def build_loaders(manifest: str, cfg: TrainConfig, shuffle: bool = True) -> DataLoader:
    ds = DualViewDataset(manifest)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, pin_memory=True, drop_last=shuffle)


def main() -> None:
    ap = argparse.ArgumentParser(description="DASF-Net -- core methodology")
    ap.add_argument("--smoke-test", action="store_true",
                    help="run every code path on random tensors; needs no data")
    ap.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")
    ap.add_argument("--source_manifest", type=str, default=None,
                    help="labelled source corpus (WaveFake, English)")
    ap.add_argument("--target_manifest", type=str, default=None,
                    help="target corpus (BanglaFake); labels unused in training")
    ap.add_argument("--seed", type=int, default=TRAIN_CFG.seed)
    ap.add_argument("--batch_size", type=int, default=TRAIN_CFG.batch_size)
    ap.add_argument("--out", type=str, default="dasfnet.pt")
    args = ap.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if not args.source_manifest or not args.target_manifest:
        ap.error("--source_manifest and --target_manifest are required "
                 "(or use --smoke-test)")

    set_seed(args.seed)
    cfg = TrainConfig(batch_size=args.batch_size, seed=args.seed)
    print(f"device: {cfg.device}   seed: {cfg.seed}")

    src = build_loaders(args.source_manifest, cfg, shuffle=True)
    tgt = build_loaders(args.target_manifest, cfg, shuffle=True)
    tgt_eval = build_loaders(args.target_manifest, cfg, shuffle=False)

    model = DASFNet(MODEL_CFG)

    if args.stage in ("1", "all"):
        model = stage1_warmup(model, src, cfg)
    if args.stage in ("2", "all"):
        model = stage2_align(model, src, tgt, cfg)
    if args.stage in ("3", "all"):
        model, _ = stage3_harden(model, src, tgt, cfg, ATTACK_CFG)

    torch.save(model.state_dict(), args.out)
    print(f"\nsaved -> {args.out}")

    print("\n=== zero-shot target evaluation (clean) ===")
    print(evaluate(model, tgt_eval, cfg.device))
    for eps in ATTACK_CFG.epsilons:
        print(f"FGSM eps={eps}: {evaluate(model, tgt_eval, cfg.device, 'fgsm', eps)}")
        print(f"PGD  eps={eps}: {evaluate(model, tgt_eval, cfg.device, 'pgd', eps)}")
    print("\n=== gate analysis (E5) ===")
    print(gate_statistics(model, tgt_eval, cfg.device))


if __name__ == "__main__":
    main()
