# DASF-Net — project README

A plain-language guide to what this paper proposes, why, and how to defend it.
Read this before the meeting; the paper itself is [main.pdf](main.pdf).

---

## 1. The one-sentence version

> Audio deepfake detectors break in two ways — they fail on new languages, and
> they fail under adversarial noise. **We argue both are the same kind of
> failure, and we fix both with one architecture that carries two adversaries
> on a single shared embedding.**

That "both failures are the same problem" claim is the intellectual core of the
paper. Everything else is machinery in service of it.

---

## 2. Where this comes from

The project sits on top of two existing papers, both in this folder.

| Paper | What it gave us | What it did *not* do |
|---|---|---|
| **BanglaFake** (`2505.10885v1.pdf`) — Fahad et al., Univ. of Dhaka, 2025 | The **data**: first public Bengali deepfake audio corpus. 12,260 real + 13,260 fake (VITS end-to-end TTS), 22.05 kHz. | Never trained a detector. Only measured audio quality (MOS 3.40 naturalness / 4.01 intelligibility) and showed a t-SNE plot where real and fake overlap heavily. |
| **Mel-Spectrogram for CNN Adversarial Defense** (`CameraReady (1).pdf`) — Ahmed, Arian, Lisa, NSU | The **method**: showed that for CNNs, dense log-Mel beats sparse LFCC (0.034 vs 0.053 EER, 36% better, across 7 WaveFake vocoders), and that adaptive FGSM/PGD training recovers robustness (FGSM 0.70 → 0.107, PGD 0.90 → 0.154). | English only. One dataset. Treated Mel and LFCC as either/or. Explicitly listed "hybrid representations" as future work. |

A third paper we cite but don't own, **"Zero-Shot to Zero-Lies"** (ICCIT 2025),
was the first to actually benchmark detectors on BanglaFake. Its best numbers
are weak — 46.20% EER zero-shot, 24.35% EER fine-tuned (ResNet18) — and its
authors name cross-lingual transfer and adversarial robustness as open
problems. **Those two open problems are exactly what we attack.**

### Why this dataset pairing is interesting

WaveFake's fakes are 7 **GAN/flow vocoders** that resynthesise a waveform from
real speech features. BanglaFake's fakes come from **one VITS end-to-end TTS**
system that generates speech straight from text. So going English → Bengali
changes **the language *and* the synthesis paradigm at the same time**. That is
a harder and more realistic test than either alone.

---

## 3. The problem, stated precisely

When a detector transfers badly to a new language, the usual reading is "it
needs more data." We claim that reading is wrong. **The detector learned the
wrong thing.** Its features encode *which corpus, which speaker, which
language* produced the audio — not *what a synthesiser did to it*.

Now notice the adversarial-attack failure. FGSM and PGD work by finding a tiny
direction in input space that flips the decision. That is also a case of the
model relying on something fragile and incidental.

**Both failures = the model latched onto a nuisance direction in feature
space.** And in machine learning, the standard cure for "stop using this
direction" is to attach an *adversary* that punishes exactly that direction.

- Nuisance = language/corpus identity → cure = **domain-adversarial training**
  (a gradient reversal layer).
- Nuisance = locally exploitable gradient → cure = **FGSM/PGD adversarial
  training**.

Nobody has put both on the same embedding for audio deepfake detection. That's
the gap.

---

## 4. DASF-Net, component by component

**DASF-Net = Dual-Adversarial Spectral-Fusion Network.** See Fig. 1 (whole
method), Fig. 2 (the network), Fig. 3 (the training schedule).

### 4.1 High-Band-Preserving preprocessing (HBP)
We keep audio at its native **22.05 kHz** instead of downsampling to 16 kHz.

*Why it matters:* 16 kHz caps you at an 8 kHz Nyquist ceiling and throws away
the 8–11 kHz band — which is exactly where vocoder and TTS artifacts live, and
exactly where GAN vocoders and VITS most differ. Our own prior paper
downsampled; for a within-English study that was survivable, but for a
cross-*paradigm* claim it deletes the evidence. **This is ablated (experiment
E4), not just asserted.**

### 4.2 Dual-branch spectral front-end
The same 4-second clip is turned into **two views**:

- **Branch-S (sparse cepstral)** — LFCC: linear filterbank (20) → DCT → 20
  coefficients → Δ + ΔΔ → `(60, T)`
- **Branch-D (dense spectral)** — log-Mel: 128 mel bins → log-power, *no DCT* →
  Δ + ΔΔ → `(3, 128, T)`

Both share identical STFT settings (n_fft 2048, hop 512), so any difference
between them is caused by the filterbank alone and nothing else. Both are
resized to 299×299×3.

### 4.3 Siamese-shared Xception trunk
Each branch gets a tiny **adapter stem** (2 conv layers, ~0.06 M params), then
**both go through the *same* Xception backbone with tied weights** (~23 M,
ImageNet-pretrained).

*Why shared and not two separate backbones:* two independent backbones would
each build a private feature space, and the fusion module would then have to
reconcile two unrelated geometries. One trunk **forces both views into a single
artifact space**, and costs 0.12 M extra parameters instead of a second 23 M
model. This is a deliberate design commitment — expect to be asked about it.

### 4.4 CASF — Cross-Attention Spectral Fusion ⭐
This is the architectural centrepiece.

Each feature map becomes 100 tokens projected to 256 dimensions. Then **two-way
cross-attention**: the Mel view queries the LFCC view, and the LFCC view
queries the Mel view. Finally a **learned gate** mixes them:

```
g = σ(W_g · [ĥ_D ; ĥ_S])
z = g ⊙ ĥ_D + (1 − g) ⊙ ĥ_S        →   z ∈ ℝ²⁵⁶
```

*Why a gate instead of just concatenating:* concatenation asserts both views
always matter equally. The gate lets the network decide **per utterance and per
dimension** which representation carries the artifact.

*Bonus — this makes the model interpretable.* The distribution of `g` across a
test set is a direct, quantitative answer to: *"do GAN vocoders and end-to-end
TTS leave their traces in different representations?"* We report that as a
result in its own right (experiment E5). Our previous paper could only report
an accuracy delta; this reports a *mechanism*.

### 4.5 Three heads on one embedding
| Head | Predicts | Role |
|---|---|---|
| `h_det` | real vs. deepfake | the actual task, loss `L_det` |
| `h_syn` | {real, GAN-vocoder, E2E-TTS} | auxiliary; organises the embedding around *mechanism* |
| `h_lang` | English vs. Bengali | **behind a gradient reversal layer** |

Note `h_syn` predicts the **family**, not the specific generator. Predicting the
exact generator would invite memorising generator identity — which is precisely
the failure that breaks cross-vocoder transfer.

**The Gradient Reversal Layer (GRL)** is the trick worth understanding well.
Forward pass: it does nothing (identity). Backward pass: it multiplies the
gradient by −λ_d. So `h_lang` tries as hard as it can to tell English from
Bengali, while the *shared trunk is being pushed to make that impossible*. At
convergence the embedding keeps what separates real from fake and discards what
separates Bengali from English.

### 4.6 The dual-adversarial objective

```
L_total = L_det(x) + λ_adv·L_det(x + δ) + λ_syn·L_syn − λ_d·L_lang
```

Two adversaries of genuinely different kinds:

- **Perturbation adversary — local.** Searches a small neighbourhood of each
  input for a direction that flips the decision.
- **Domain adversary — global.** Searches the whole embedding for any direction
  that reveals which corpus the audio came from.

Our hypothesis is that they help each other: an embedding stripped of language
identity offers fewer spurious directions for an attack to exploit, and an
embedding hardened against local perturbation relies less on the brittle
corpus-specific cues the domain adversary attacks. **The ablation is built to
test this, not to assume it.**

### 4.7 Three-stage training curriculum
Training all of that from scratch is unstable — a reversed gradient from an
untrained discriminator is large and meaningless, and adversarial examples
against an untrained boundary are noise. So:

| Stage | What happens | Why |
|---|---|---|
| **1. Warm-up** | Trunk frozen. Train adapters + CASF + `h_det` on labelled source only. | Get a usable fused representation before disturbing pretrained weights. |
| **2. Cross-lingual alignment** | Unfreeze trunk, attach `h_syn` and `h_lang`. Mix in **unlabelled** target audio. λ_d ramps up from 0. | Build language invariance. Target *labels are never used* — only the language tag. This keeps it honestly unsupervised on the target. |
| **3. Dual-adversarial hardening** | Add adaptive FGSM/PGD at lr 5e-5, GRL still on. | Harden a representation that is *already* language-invariant. |

**Adaptive attack sampling:** instead of a fixed FGSM:PGD ratio, we sample each
minibatch's attack in proportion to a momentum-smoothed estimate (m = 0.2) of
how well that attack has been working lately, keeping p_c = 1/3 of each batch
clean. Training automatically concentrates on whichever attack currently works.

---

## 5. Two methodology fixes that matter more than they sound

These are the parts a careful reviewer will respect most.

### 5.1 CCEP — Confound-Controlled Evaluation Protocol
BanglaFake has a problem nobody has flagged: **the fake side is one male
speaker, the real side is seven speakers of both genders.** A model can score
brilliantly by classifying *gender*, learning nothing about deepfakes at all.
Plus WaveFake is ~8:1 fake:real while BanglaFake is ~1:1, so accuracy is not
comparable across them. So every number we report is subject to:

1. **Speaker-disjoint splits** — no speaker in both train and test.
2. **Gender-balanced Bengali evaluation** — primary test set restricted to male
   real speech; the mixed-gender number reported separately as a diagnostic. A
   big gap between the two *is itself evidence of gender shortcutting.*
3. **Prior-matched test sets** — both resampled to 1:1 so accuracy means something.
4. **EER as the headline metric** (threshold-free), with min-tDCF and balanced
   accuracy. Raw accuracy never reported for a transfer condition without its prior.

### 5.2 A corrected adversarial threat model
Our own prior paper reports FGSM at ε ∈ {0.0005, 0.00075, 0.001} and PGD at
ε ∈ {0.1, 0.15, 0.20} — **budgets two orders of magnitude apart**. That makes
"FGSM 0.107 vs PGD 0.154" a meaningless comparison; you cannot say one attack is
stronger when they were given different budgets. We put both attacks on **one
shared ε grid** {0.001, 0.005, 0.010}, define them in the normalised feature
domain, and report every robustness number per ε. The paper says this
explicitly, in print.

*(Self-correcting your own group's prior paper is a good look in a defence, not
a bad one. Own it.)*

---

## 6. What we will actually run

Three research questions:

- **RQ1 (representation)** — does the dense-over-sparse advantage hold on
  Bengali VITS audio, and does *learned fusion* beat either view alone?
- **RQ2 (cross-lingual)** — how much of the zero-shot English↔Bengali gap does
  the dual-adversarial objective close?
- **RQ3 (robustness transfer)** — does adversarial robustness learned in one
  language survive transfer to the other?

Five experiments:

| # | Experiment | Answers |
|---|---|---|
| E1 | In-language baselines: Branch-S only / Branch-D only / full CASF, per corpus | RQ1 |
| E2 | Zero-shot transfer both directions, with and without the language adversary; per-vocoder breakdown; ≤10% few-shot variant | RQ2 |
| E3 | Clean/FGSM/PGD EER per ε, in-domain vs. transferred robustness | RQ3 |
| E4 | **Ablation** — remove CASF / shared trunk / `h_syn` / GRL / adversarial stage / HBP one at a time | attribution |
| E5 | Gate analysis — distribution of `g` per generator family | interpretability |

**E4 is the experiment that makes this a paper rather than a demo.** Without it,
any improvement is unattributed; with it, we can say *which* component earned it.

All runs: 3 random seeds, mean ± std. No single-seed results.

---

## 7. Anticipated questions, and honest answers

**"Isn't this just applying an existing pipeline to a new dataset?"**
No — that was the *previous* draft, and it was a fair criticism of it. That
version reused the Mel/Xception pipeline unchanged and only measured the
transfer gap. This version adds a fusion module, a second adversary, a shared-
trunk two-branch encoder, and a training curriculum. It *closes* the gap rather
than reporting it.

**"Why not just fine-tune a large pretrained speech model like Wav2Vec2?"**
That's the Zero-Shot to Zero-Lies baseline, and it reached 24.35% EER
fine-tuned. Our claim is architectural, not about scale: what matters is
*removing language information* from the embedding, which no amount of
pretraining does on its own.

**"Does the gradient reversal layer actually converge?"**
It's a known-unstable component — that is precisely why it is introduced in
Stage 2 rather than Stage 1, with λ_d ramped from 0 by the standard DANN
schedule rather than fixed. If it still destabilises, that's a reportable
negative result about the method, not a bug to hide.

**"One Bengali generator. Can you claim generalisation?"**
No, and we say so in the Limitations section. With one VITS system we can test
cross-*paradigm* transfer but **not** cross-vocoder generalisation within
Bengali the way WaveFake's 7-generator leave-one-out does for English. We do not
make that claim. Adding a Bengali FastSpeech2 or Tacotron2 is the stated next step.

**"What's the cost?"**
The shared trunk runs twice per utterance (≈2× forward/backward vs. single
branch), and Stage 3 adds the usual adversarial-training overhead with PGD at
K=10 dominating. Inference stays real-time on one modern GPU.

**"What if the dual-adversarial hypothesis is wrong?"**
Then E4 shows the GRL and the adversarial stage don't help each other, and we
report that. The experimental design is built to be informative either way —
which is the point of running the ablation instead of only the full model.

---

## 8. Status — be honest about this

✅ Done: architecture, methodology, algorithms (both in pseudocode), experimental
design, threat model, evaluation protocol, three figures, compiled paper.

❌ Not done: **every experiment.** There are no results, no numbers, no trained
model. The paper deliberately has no Results section and says so in its own
abstract and header comment.

This is a **design/methodology paper awaiting its empirical study.** Present it
that way — claiming otherwise is the fastest way to lose the room.

---

## 9. Files

| File | What it is |
|---|---|
| `main.tex` | Paper source (IEEE conference format) |
| `main.pdf` | Compiled paper, 8 pages |
| `main.tex.bak` | The earlier draft, before the DASF-Net redesign |
| `fig1-methodology.drawio{,.pdf,.png}` | Fig. 1 — end-to-end method |
| `fig2-dasfnet.drawio{,.pdf,.png}` | Fig. 2 — the network in detail |
| `fig3-training.drawio{,.pdf,.png}` | Fig. 3 — three-stage curriculum |
| `2505.10885v1.pdf` | Source paper 1 — BanglaFake |
| `CameraReady (1).pdf` | Source paper 2 — Mel-Spectrogram / Xception |

**Build:** `pdflatex main.tex` twice. No bibtex pass needed.

**Editing figures:** open the `.drawio` files in draw.io Desktop. Re-export with:
```
drawio -x -f pdf -e -b 8 -o fig1-methodology.drawio.pdf fig1-methodology.drawio
```

### ⚠️ References are currently switched OFF

The reference list is suppressed in this draft. Every `\cite{}` key is still in
the source and nothing was deleted. To bring references back:

1. Delete the line `\renewcommand{\cite}[1]{\unskip}` near the top of `main.tex`.
2. Delete the `\iffalse` and `\fi` lines wrapping `\begin{thebibliography}` at
   the end of `main.tex`.
3. Run `pdflatex` twice.

**Turn them back on before any real submission** — a paper making priority
claims ("first to...", "no prior work has...") without citations will not pass
review.

### Note on length
8 pages without results. Once the Results section is written this will overrun a
typical 6-page conference limit. Related Work and Limitations are the
compressible sections.
