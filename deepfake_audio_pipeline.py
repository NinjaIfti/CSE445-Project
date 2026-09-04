"""
deepfake_audio_pipeline.py
===========================================================================
Single-file methodology for audio deepfake detection, extending:

  - "Mel-Spectrogram Representations for CNN Based Adversarial Defense for
     Audio Deepfake Detection" (Xception CNN, WaveFake, LFCC vs Mel-Spec,
     FGSM/PGD adversarial training)

into the Bengali BanglaFake dataset, via three experiments:

  Experiment 1 - In-language baseline on BanglaFake (Mel vs LFCC, Xception)
  Experiment 2 - Cross-lingual / cross-architecture zero-shot transfer
                 (WaveFake <-> BanglaFake, both directions; optional
                 few-shot fine-tuning)
  Experiment 3 - Cross-lingual adversarial robustness (does FGSM/PGD
                 adversarial training on one language/dataset transfer
                 robustness to the other?)

WaveFake (English, LJSpeech single speaker) covers 7 GAN-based vocoders
(MelGAN, MelGAN-L, PWG, MB-MelGAN, FB-MelGAN, HiFi-GAN, WaveGlow).
BanglaFake (Bengali, SUST TTS Corpus + Common Voice speakers) covers a
single VITS-based (flow/VAE, non-GAN) synthesis system. Because VITS is
architecturally distinct from all 7 WaveFake vocoders, cross-dataset
transfer is simultaneously a cross-lingual AND cross-architecture
generalization test, even without additional Bengali vocoders.

Dependencies (install with pip, --break-system-packages if needed):
    pip install torch torchvision timm librosa soundfile scikit-learn \
                numpy pandas tqdm --break-system-packages

Usage:
    python deepfake_audio_pipeline.py --experiment 1 --dataset banglafake \
        --manifest /path/to/banglafake_manifest.csv

    python deepfake_audio_pipeline.py --experiment 2 --direction en2bn \
        --train_manifest /path/to/wavefake_manifest.csv \
        --test_manifest /path/to/banglafake_manifest.csv

    python deepfake_audio_pipeline.py --experiment 3 --direction en2bn \
        --train_manifest /path/to/wavefake_manifest.csv \
        --test_manifest /path/to/banglafake_manifest.csv

Manifest CSV format (LJSpeech-style, one row per utterance):
    filepath,label,vocoder,language
    /data/wavefake/ljspeech_melgan/LJ001-0001.wav,1,melgan,en
    /data/wavefake/real/LJ001-0002.wav,0,real,en
    /data/banglafake/fake/0001.wav,1,vits,bn
    /data/banglafake/real/0002.wav,0,real,bn

label: 0 = bonafide/real, 1 = deepfake/spoof
vocoder: synthesis method name ("real" for genuine audio)
language: "en" or "bn" (used only for bookkeeping / experiment filters)
===========================================================================
"""

import argparse
import copy
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import librosa
except ImportError as e:
    raise ImportError(
        "librosa is required for feature extraction: pip install librosa"
    ) from e

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for the pretrained Xception backbone: pip install timm"
    ) from e

from sklearn.metrics import roc_curve

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ===========================================================================
# 1. CONFIG
# ===========================================================================

@dataclass
class AudioConfig:
    sample_rate: int = 16000          # both datasets resampled to this
    target_duration_s: float = 4.0    # normalize all clips to 4s
    silence_gap_s: float = 2.0        # trim silences longer than this
    silence_db_threshold: float = 40.0

    # STFT (shared by LFCC and Mel pipelines)
    n_fft: int = 2048
    hop_length: int = 512

    # LFCC pipeline
    lfcc_n_filters: int = 20
    lfcc_n_coeff: int = 20
    lfcc_fmin: float = 0.0
    lfcc_fmax: float = 8000.0

    # Mel-Spectrogram pipeline
    mel_n_mels: int = 128

    # CNN input size (Xception expects 299x299x3)
    cnn_input_size: int = 299


@dataclass
class TrainConfig:
    batch_size: int = 128
    phase1_epochs: int = 5            # train final layer only
    phase2_epochs: int = 15           # fine-tune entire network
    lr_phase1: float = 1e-4
    lr_phase2: float = 1e-4
    label_smoothing: float = 0.1
    mixed_precision: bool = True
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class AdversarialConfig:
    fgsm_epsilons: Tuple[float, ...] = (0.0005, 0.00075, 0.001)
    pgd_epsilons: Tuple[float, ...] = (0.1, 0.15, 0.20)
    pgd_steps: int = 10
    adaptive_epochs: int = 10
    adaptive_lr: float = 5e-5
    momentum: float = 0.2             # attack-sampling momentum (m)
    clean_sample_prob: float = 1.0 / 3.0  # fraction of clean samples kept (p)


AUDIO_CFG = AudioConfig()
TRAIN_CFG = TrainConfig()
ADV_CFG = AdversarialConfig()


# ===========================================================================
# 2. PREPROCESSING
# ===========================================================================

def load_and_preprocess_audio(filepath: str, cfg: AudioConfig = AUDIO_CFG) -> np.ndarray:
    """Resample to 16kHz mono, trim long silences, normalize duration to 4s.

    Mirrors the WaveFake-paper preprocessing so BOTH datasets land in an
    identical audio representation before feature extraction -- this is
    what makes cross-lingual / cross-dataset transfer experiments valid.
    """
    y, sr = librosa.load(filepath, sr=cfg.sample_rate, mono=True)

    # Trim silence gaps longer than `silence_gap_s` at the given dB threshold.
    intervals = librosa.effects.split(y, top_db=cfg.silence_db_threshold)
    if len(intervals) > 0:
        chunks = []
        for start, end in intervals:
            chunks.append(y[start:end])
        y = np.concatenate(chunks) if chunks else y

    target_len = int(cfg.target_duration_s * cfg.sample_rate)
    if len(y) >= target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")

    return y.astype(np.float32)


# ===========================================================================
# 3. FEATURE EXTRACTION -- LFCC and Mel-Spectrogram pipelines
# ===========================================================================

def _add_deltas(feat: np.ndarray) -> np.ndarray:
    """Stack static + delta + delta-delta along a new leading axis."""
    delta = librosa.feature.delta(feat, order=1)
    delta2 = librosa.feature.delta(feat, order=2)
    return np.stack([feat, delta, delta2], axis=0)  # (3, n_coeff/n_mel, time)


def _resize_to_cnn_input(feat: np.ndarray, size: int = AUDIO_CFG.cnn_input_size) -> np.ndarray:
    """Resize a (3, F, T) feature tensor to (3, size, size) via interpolation."""
    tensor = torch.from_numpy(feat).unsqueeze(0)  # (1, 3, F, T)
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return resized.squeeze(0).numpy()  # (3, size, size)


def extract_lfcc(y: np.ndarray, cfg: AudioConfig = AUDIO_CFG) -> np.ndarray:
    """LFCC pipeline: STFT -> linear filterbank -> DCT -> 20 coeffs.

    First/second derivatives yield a (60, time) representation which is
    then resized to (299, 299, 3) -- here returned as (3, 299, 299) for
    PyTorch's channel-first convention.
    """
    stft = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length)) ** 2
    linear_fb = librosa.filters.linear(
        sr=cfg.sample_rate, n_fft=cfg.n_fft,
        n_filter=cfg.lfcc_n_filters, fmin=cfg.lfcc_fmin, fmax=cfg.lfcc_fmax,
    )
    linear_spec = np.dot(linear_fb, stft)
    log_linear_spec = librosa.power_to_db(linear_spec + 1e-10)
    lfcc = librosa.feature.mfcc(
        S=log_linear_spec, n_mfcc=cfg.lfcc_n_coeff
    )  # DCT-based cepstral coefficients, (20, time)

    feat = _add_deltas(lfcc)  # (3, 20, time) -- treated as (channels=3, coeff, time)
    # Match paper's (60, time) -> resize: fold the 3-deriv stack into height.
    feat = feat.reshape(1, cfg.lfcc_n_coeff * 3, feat.shape[-1])
    feat = np.repeat(feat, 3, axis=0)  # replicate to 3 "channels" for CNN resize step
    return _resize_to_cnn_input(feat)


def extract_mel_spectrogram(y: np.ndarray, cfg: AudioConfig = AUDIO_CFG) -> np.ndarray:
    """Mel-Spectrogram pipeline: STFT -> mel filterbank -> log power (no DCT).

    Preserves dense spectral information. Deltas form a (3, 128, time)
    tensor, resized to (299, 299, 3) -- returned as (3, 299, 299).
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, n_mels=cfg.mel_n_mels,
    )
    log_mel = librosa.power_to_db(mel + 1e-10)  # (128, time)
    feat = _add_deltas(log_mel)  # (3, 128, time)
    return _resize_to_cnn_input(feat)


FEATURE_EXTRACTORS = {
    "lfcc": extract_lfcc,
    "mel": extract_mel_spectrogram,
}


# ===========================================================================
# 4. DATASET
# ===========================================================================

class DeepfakeAudioDataset(Dataset):
    """Loads a manifest CSV (filepath,label,vocoder,language) and extracts
    the requested feature representation on the fly.

    Set `precompute=True` to cache features to disk (recommended for real
    training runs -- feature extraction is the bottleneck otherwise).
    """

    def __init__(
        self,
        manifest_path: str,
        feature_type: str = "mel",
        cfg: AudioConfig = AUDIO_CFG,
        vocoder_filter: Optional[List[str]] = None,
        language_filter: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        assert feature_type in FEATURE_EXTRACTORS, f"Unknown feature_type: {feature_type}"
        df = pd.read_csv(manifest_path)
        if vocoder_filter is not None:
            df = df[df["vocoder"].isin(vocoder_filter)]
        if language_filter is not None:
            df = df[df["language"].isin(language_filter)]
        self.df = df.reset_index(drop=True)
        self.feature_type = feature_type
        self.extractor = FEATURE_EXTRACTORS[feature_type]
        self.cfg = cfg
        self.cache_dir = cache_dir
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)

    def _cache_path(self, idx: int) -> Optional[str]:
        if self.cache_dir is None:
            return None
        base = os.path.splitext(os.path.basename(self.df.iloc[idx]["filepath"]))[0]
        return os.path.join(self.cache_dir, f"{base}_{self.feature_type}.npy")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        cache_path = self._cache_path(idx)

        if cache_path is not None and os.path.exists(cache_path):
            feat = np.load(cache_path)
        else:
            y = load_and_preprocess_audio(row["filepath"], self.cfg)
            feat = self.extractor(y, self.cfg)
            if cache_path is not None:
                np.save(cache_path, feat)

        label = int(row["label"])
        return torch.from_numpy(feat).float(), torch.tensor(label, dtype=torch.long)


def make_loader(dataset: Dataset, shuffle: bool, cfg: TrainConfig = TRAIN_CFG) -> DataLoader:
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )


# ===========================================================================
# 5. MODEL -- Xception, ImageNet-pretrained, binary head
# ===========================================================================

def build_xception(num_classes: int = 2) -> nn.Module:
    model = timm.create_model("xception", pretrained=True, num_classes=num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Phase 1: train only the final classification layer."""
    for name, param in model.named_parameters():
        param.requires_grad = ("fc" in name) or ("classifier" in name) or ("head" in name)


def unfreeze_all(model: nn.Module) -> None:
    """Phase 2: fine-tune the entire network."""
    for param in model.parameters():
        param.requires_grad = True


# ===========================================================================
# 6. ADVERSARIAL ATTACKS -- FGSM, PGD
# ===========================================================================

def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = x + epsilon * grad.sign()
    return x_adv.detach()


def pgd_attack(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor,
    epsilon: float, steps: int = ADV_CFG.pgd_steps,
) -> torch.Tensor:
    alpha = epsilon / 5.0  # matches paper: alpha = epsilon / 5
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        perturbation = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = (x_orig + perturbation).detach()
    return x_adv


def sample_attack(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, attack_name: str,
    adv_cfg: AdversarialConfig = ADV_CFG,
) -> torch.Tensor:
    if attack_name == "fgsm":
        eps = random.choice(adv_cfg.fgsm_epsilons)
        return fgsm_attack(model, x, y, eps)
    elif attack_name == "pgd":
        eps = random.choice(adv_cfg.pgd_epsilons)
        return pgd_attack(model, x, y, eps, adv_cfg.pgd_steps)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


# ===========================================================================
# 7. TRAINING -- two-phase standard training, adaptive adversarial training
# ===========================================================================

def train_standard(
    model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader],
    cfg: TrainConfig = TRAIN_CFG,
) -> nn.Module:
    """Phase 1 (final layer only) then Phase 2 (full fine-tune), matching
    the WaveFake-paper protocol: Adam, label-smoothed cross-entropy, mixed
    precision.
    """
    device = torch.device(cfg.device)
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)

    def run_epochs(n_epochs: int, lr: float):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
        for epoch in range(n_epochs):
            model.train()
            total_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * x.size(0)
            avg_loss = total_loss / len(train_loader.dataset)
            msg = f"  epoch {epoch + 1}/{n_epochs} - train_loss={avg_loss:.4f}"
            if val_loader is not None:
                acc, eer = evaluate(model, val_loader, cfg)
                msg += f" - val_acc={acc:.4f} - val_eer={eer:.4f}"
            print(msg)

    print("Phase 1: training final layer only")
    freeze_backbone(model)
    run_epochs(cfg.phase1_epochs, cfg.lr_phase1)

    print("Phase 2: fine-tuning entire network")
    unfreeze_all(model)
    run_epochs(cfg.phase2_epochs, cfg.lr_phase2)

    return model


def train_adaptive_adversarial(
    model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader],
    train_cfg: TrainConfig = TRAIN_CFG, adv_cfg: AdversarialConfig = ADV_CFG,
) -> nn.Module:
    """Adaptive adversarial fine-tuning with momentum-weighted dynamic
    attack sampling (m) and a fixed clean-sample retention fraction (p),
    matching the WaveFake-paper recipe.
    """
    device = torch.device(train_cfg.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=adv_cfg.adaptive_lr)

    # Running success rates for FGSM / PGD, used to weight sampling.
    attack_success_rate = {"fgsm": 0.5, "pgd": 0.5}

    for epoch in range(adv_cfg.adaptive_epochs):
        model.train()
        total_loss = 0.0
        n_seen = 0
        epoch_success = {"fgsm": [], "pgd": []}

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            use_clean = random.random() < adv_cfg.clean_sample_prob
            if use_clean:
                x_input = x
            else:
                # Weighted choice between FGSM / PGD based on current success rate.
                total = attack_success_rate["fgsm"] + attack_success_rate["pgd"]
                p_fgsm = attack_success_rate["fgsm"] / total if total > 0 else 0.5
                attack_name = "fgsm" if random.random() < p_fgsm else "pgd"
                model.eval()
                x_input = sample_attack(model, x, y, attack_name, adv_cfg)
                model.train()

                with torch.no_grad():
                    preds = model(x_input).argmax(dim=1)
                    success = (preds != y).float().mean().item()
                epoch_success[attack_name].append(success)

            logits = model(x_input)
            loss = F.cross_entropy(logits, y, label_smoothing=train_cfg.label_smoothing)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            n_seen += x.size(0)

        # Momentum-smoothed update of attack success rates.
        for name in ("fgsm", "pgd"):
            if epoch_success[name]:
                new_rate = float(np.mean(epoch_success[name]))
                attack_success_rate[name] = (
                    adv_cfg.momentum * attack_success_rate[name]
                    + (1 - adv_cfg.momentum) * new_rate
                )

        avg_loss = total_loss / max(n_seen, 1)
        msg = f"  [adv] epoch {epoch + 1}/{adv_cfg.adaptive_epochs} - loss={avg_loss:.4f}"
        if val_loader is not None:
            clean_acc, clean_eer = evaluate(model, val_loader, train_cfg)
            fgsm_eer = evaluate_under_attack(model, val_loader, "fgsm", train_cfg, adv_cfg)
            pgd_eer = evaluate_under_attack(model, val_loader, "pgd", train_cfg, adv_cfg)
            msg += (
                f" - clean_eer={clean_eer:.4f} - fgsm_eer={fgsm_eer:.4f} - pgd_eer={pgd_eer:.4f}"
            )
        print(msg)

    return model


# ===========================================================================
# 8. EVALUATION -- accuracy, EER, robustness under attack
# ===========================================================================

def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Equal Error Rate from bonafide/spoof scores (higher score = more
    likely deepfake, matching label convention 1=fake)."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg: TrainConfig = TRAIN_CFG) -> Tuple[float, float]:
    device = torch.device(cfg.device)
    model = model.to(device).eval()
    all_labels, all_scores, correct, total = [], [], 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)[:, 1]  # P(deepfake)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_labels.append(y.cpu().numpy())
        all_scores.append(probs.cpu().numpy())

    acc = correct / max(total, 1)
    eer = compute_eer(np.concatenate(all_labels), np.concatenate(all_scores))
    return acc, eer


def evaluate_under_attack(
    model: nn.Module, loader: DataLoader, attack_name: str,
    train_cfg: TrainConfig = TRAIN_CFG, adv_cfg: AdversarialConfig = ADV_CFG,
) -> float:
    device = torch.device(train_cfg.device)
    model = model.to(device).eval()
    all_labels, all_scores = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = sample_attack(model, x, y, attack_name, adv_cfg)
        with torch.no_grad():
            logits = model(x_adv)
            probs = F.softmax(logits, dim=1)[:, 1]
        all_labels.append(y.cpu().numpy())
        all_scores.append(probs.cpu().numpy())

    return compute_eer(np.concatenate(all_labels), np.concatenate(all_scores))


# ===========================================================================
# 9. EXPERIMENT ORCHESTRATION
# ===========================================================================

def build_train_val_split(manifest_path: str, feature_type: str, val_frac: float = 0.2,
                           **filter_kwargs) -> Tuple[Dataset, Dataset]:
    full = DeepfakeAudioDataset(manifest_path, feature_type=feature_type, **filter_kwargs)
    n_val = int(len(full) * val_frac)
    n_train = len(full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    return train_ds, val_ds


def experiment_1_inlanguage_baseline(manifest_path: str) -> None:
    """In-language baseline on BanglaFake (or any single dataset): Mel vs
    LFCC, Xception CNN. Reproduces the original paper's core ablation on
    the new dataset."""
    print("=" * 70)
    print("EXPERIMENT 1: In-language baseline (Mel vs LFCC)")
    print("=" * 70)
    for feature_type in ("lfcc", "mel"):
        print(f"\n--- Feature: {feature_type.upper()} ---")
        train_ds, val_ds = build_train_val_split(manifest_path, feature_type)
        train_loader = make_loader(train_ds, shuffle=True)
        val_loader = make_loader(val_ds, shuffle=False)

        model = build_xception(num_classes=2)
        model = train_standard(model, train_loader, val_loader)
        acc, eer = evaluate(model, val_loader)
        print(f"[RESULT] {feature_type.upper()} -> acc={acc:.4f}  EER={eer:.4f}")


def experiment_2_cross_lingual_transfer(
    train_manifest: str, test_manifest: str,
    per_vocoder: bool = False, finetune_frac: float = 0.0,
) -> None:
    """Train on one dataset/language, zero-shot test on the other.

    Set `per_vocoder=True` when testing on WaveFake to reproduce the
    original paper's per-vocoder breakdown table under transfer.
    Set `finetune_frac > 0` to additionally report few-shot fine-tuning
    results on a small slice of the target-domain data.
    """
    print("=" * 70)
    print("EXPERIMENT 2: Cross-lingual / cross-architecture zero-shot transfer")
    print("=" * 70)

    for feature_type in ("lfcc", "mel"):
        print(f"\n--- Feature: {feature_type.upper()} ---")
        train_ds, train_val_ds = build_train_val_split(train_manifest, feature_type)
        train_loader = make_loader(train_ds, shuffle=True)
        train_val_loader = make_loader(train_val_ds, shuffle=False)

        model = build_xception(num_classes=2)
        model = train_standard(model, train_loader, train_val_loader)

        # Zero-shot evaluation on the target domain.
        if per_vocoder:
            test_df = pd.read_csv(test_manifest)
            vocoders = sorted(test_df["vocoder"].unique())
            for voc in vocoders:
                ds = DeepfakeAudioDataset(
                    test_manifest, feature_type=feature_type, vocoder_filter=[voc, "real"]
                )
                loader = make_loader(ds, shuffle=False)
                acc, eer = evaluate(model, loader)
                print(f"[RESULT] zero-shot -> vocoder={voc:20s} acc={acc:.4f}  EER={eer:.4f}")
        else:
            test_ds = DeepfakeAudioDataset(test_manifest, feature_type=feature_type)
            test_loader = make_loader(test_ds, shuffle=False)
            acc, eer = evaluate(model, test_loader)
            print(f"[RESULT] zero-shot transfer -> acc={acc:.4f}  EER={eer:.4f}")

        # Optional: few-shot fine-tuning on a small slice of target data.
        if finetune_frac > 0:
            ft_ds, ft_val_ds = build_train_val_split(
                test_manifest, feature_type, val_frac=0.5
            )
            n_ft = int(len(ft_ds) * finetune_frac / (1 - 0.5))
            ft_subset, _ = torch.utils.data.random_split(
                ft_ds, [n_ft, len(ft_ds) - n_ft],
                generator=torch.Generator().manual_seed(SEED),
            )
            ft_loader = make_loader(ft_subset, shuffle=True)
            ft_val_loader = make_loader(ft_val_ds, shuffle=False)
            ft_model = copy.deepcopy(model)
            unfreeze_all(ft_model)
            ft_cfg = TrainConfig(phase1_epochs=0, phase2_epochs=5)
            ft_model = train_standard(ft_model, ft_loader, ft_val_loader, cfg=ft_cfg)
            acc, eer = evaluate(ft_model, ft_val_loader)
            print(f"[RESULT] few-shot fine-tuned ({finetune_frac:.0%} target data) "
                  f"-> acc={acc:.4f}  EER={eer:.4f}")


def experiment_3_cross_lingual_adversarial(
    train_manifest: str, test_manifest: str, feature_type: str = "mel",
) -> None:
    """Does adversarial robustness transfer across languages/datasets?

    1. Adversarially train on the source domain, attack-evaluate on the
       target domain (transferred robustness).
    2. Adversarially train directly on the target domain (in-domain
       robustness) for comparison.
    """
    print("=" * 70)
    print("EXPERIMENT 3: Cross-lingual adversarial robustness")
    print(f"(feature = {feature_type.upper()})")
    print("=" * 70)

    # --- Source-domain adversarial training ---
    src_train_ds, src_val_ds = build_train_val_split(train_manifest, feature_type)
    src_train_loader = make_loader(src_train_ds, shuffle=True)
    src_val_loader = make_loader(src_val_ds, shuffle=False)

    print("\n[1/2] Training clean model on source domain, then adversarially fine-tuning...")
    model = build_xception(num_classes=2)
    model = train_standard(model, src_train_loader, src_val_loader)
    adv_model = train_adaptive_adversarial(model, src_train_loader, src_val_loader)

    tgt_ds = DeepfakeAudioDataset(test_manifest, feature_type=feature_type)
    tgt_loader = make_loader(tgt_ds, shuffle=False)

    clean_acc, clean_eer = evaluate(adv_model, tgt_loader)
    fgsm_eer = evaluate_under_attack(adv_model, tgt_loader, "fgsm")
    pgd_eer = evaluate_under_attack(adv_model, tgt_loader, "pgd")
    print(f"[RESULT] transferred robustness (src-adv-trained -> tgt-tested) "
          f"clean_eer={clean_eer:.4f}  fgsm_eer={fgsm_eer:.4f}  pgd_eer={pgd_eer:.4f}")

    # --- Target-domain (in-domain) adversarial training for comparison ---
    print("\n[2/2] Training clean model on target domain, then adversarially fine-tuning...")
    tgt_train_ds, tgt_val_ds = build_train_val_split(test_manifest, feature_type)
    tgt_train_loader = make_loader(tgt_train_ds, shuffle=True)
    tgt_val_loader = make_loader(tgt_val_ds, shuffle=False)

    indomain_model = build_xception(num_classes=2)
    indomain_model = train_standard(indomain_model, tgt_train_loader, tgt_val_loader)
    indomain_adv_model = train_adaptive_adversarial(indomain_model, tgt_train_loader, tgt_val_loader)

    clean_acc2, clean_eer2 = evaluate(indomain_adv_model, tgt_val_loader)
    fgsm_eer2 = evaluate_under_attack(indomain_adv_model, tgt_val_loader, "fgsm")
    pgd_eer2 = evaluate_under_attack(indomain_adv_model, tgt_val_loader, "pgd")
    print(f"[RESULT] in-domain robustness (tgt-adv-trained -> tgt-tested) "
          f"clean_eer={clean_eer2:.4f}  fgsm_eer={fgsm_eer2:.4f}  pgd_eer={pgd_eer2:.4f}")

    print("\n[SUMMARY] Compare the two [RESULT] blocks above: a small gap between "
          "transferred and in-domain robustness means adversarial robustness "
          "generalizes across language/architecture; a large gap means it doesn't.")


# ===========================================================================
# 10. CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--manifest", type=str, default=None,
                         help="Manifest CSV for Experiment 1")
    parser.add_argument("--train_manifest", type=str, default=None,
                         help="Source-domain manifest CSV for Experiments 2/3")
    parser.add_argument("--test_manifest", type=str, default=None,
                         help="Target-domain manifest CSV for Experiments 2/3")
    parser.add_argument("--per_vocoder", action="store_true",
                         help="Experiment 2: break down zero-shot results by vocoder")
    parser.add_argument("--finetune_frac", type=float, default=0.0,
                         help="Experiment 2: fraction of target data for few-shot fine-tuning")
    parser.add_argument("--feature_type", type=str, default="mel", choices=["mel", "lfcc"],
                         help="Experiment 3: which feature representation to use")
    args = parser.parse_args()

    if args.experiment == 1:
        assert args.manifest, "--manifest is required for Experiment 1"
        experiment_1_inlanguage_baseline(args.manifest)
    elif args.experiment == 2:
        assert args.train_manifest and args.test_manifest, \
            "--train_manifest and --test_manifest are required for Experiment 2"
        experiment_2_cross_lingual_transfer(
            args.train_manifest, args.test_manifest,
            per_vocoder=args.per_vocoder, finetune_frac=args.finetune_frac,
        )
    elif args.experiment == 3:
        assert args.train_manifest and args.test_manifest, \
            "--train_manifest and --test_manifest are required for Experiment 3"
        experiment_3_cross_lingual_adversarial(
            args.train_manifest, args.test_manifest, feature_type=args.feature_type,
        )


if __name__ == "__main__":
    main()
