# Prioritized Research Directions
_Last updated: 2026-02-27 00:26 UTC | Run #2_

---

## 🏆 #1 — Bayesian Posterior Factual Calibration (BPFC)
_Formerly "CZEC" — now theoretically grounded via Doyle (2025)_

**Novelty score: 9.8/10** | **Feasibility: 8.5/10** | **Impact: 9.5/10**

### The Upgrade

Run #1 proposed CZEC (Confusion-Zone Epistemic Calibration) — an empirical approach.
Run #2 discovered **arXiv:2507.07586** (Doyle, Jul 2025), which **proves** that absorbing
discrete diffusion LMs implement the exact Bayesian posterior. Per-token variance from K
independent denoising passes has Spearman ρ = 0.996 with reconstruction error.

This transforms our direction: instead of *hoping* that confusion zones encode knowledge,
we now have a *theorem* that says variance = calibrated uncertainty. We're building the
first application of this theorem to factual QA and knowledge boundaries.

### The Core Research Question
> Does the per-token Bayesian posterior variance (K denoising passes, masked diffusion LM)
> constitute a calibrated epistemic signal for factual recall — and does it differentiate
> known from hallucinated facts with superior calibration to AR semantic entropy?

### Why Novel (Post-Run #2)
1. Doyle (2025) only validated on WikiText-2 perplexity — NO factual QA application
2. Nobody has connected posterior variance → knowledge boundaries
3. Nobody has compared DLM posterior variance to AR semantic entropy (Kuhn et al.)
4. Nobody has built a DLM abstention/selective-generation system from posterior variance
5. Energy of Falsehood (Feb 2026) uses EXTERNAL reconstruction stress test — we use INTERNAL generation trajectory → fundamentally different, complementary, more practically useful
6. "Diffusion uncertainty quantification factual" → 0 results on arXiv

### Formal Research Hypotheses
- **H1**: σ²_total (posterior variance over answer span) predicts factual correctness AUROC ≥ 0.75
- **H2**: σ²_total monotonically decreases with entity frequency (knowledge boundary signal)
- **H3**: K=8 passes achieves >90% of the AUROC of K=64 (compute-efficient operating point)
- **H4**: DLM posterior variance ECE ≤ AR semantic entropy ECE (better calibrated)
- **H5**: σ²_total and confusion_mass (CZEC feature) have r ≥ 0.65 (same phenomenon, two views)

### Experimental Design

```
Phase 1 — Posterior Variance Extraction (Doyle's MC estimator):
  K ∈ {4, 8, 16, 32, 64} independent masked denoising passes
  σ²_i = Var_k[argmax P_k(x_i)] per token position
  σ²_span = mean(σ²_i) over answer span positions
  Model: LLaDA-8B-Instruct (HuggingFace: GSAI-ML/LLaDA-8B-Instruct)
  Dataset: TriviaQA (500 dev questions), PopQA (stratified by entity freq)

Phase 2 — Confusion Zone Features (complementary):
  Single-pass entropy trajectory H(t), RoEC, CM
  Extract: confusion_mass, confusion_count, peak_RoEC
  Correlate with σ²_span

Phase 3 — Calibration Analysis:
  AUROC: σ²_span → factual correctness (EM)
  ECE curves: bucket by σ²_span, measure accuracy within bucket
  Compare: LLaMA-8B semantic entropy (5 samples/question), verbalized confidence

Phase 4 — Knowledge Boundary:
  PopQA grouped by log10(entity_freq): 1–2 (rare) → 4–5 (common)
  Hypothesis: σ²_span ~ 1/log(entity_freq)
```

### Expected Contributions
1. First practical calibration system for DLMs grounded in Bayesian posterior theory
2. First comparison of DLM vs AR uncertainty quantification on factual QA
3. A new meta-cognitive capability: DLMs saying "I don't know" from internal signals only
4. Connection between Doyle's theoretical MC estimator and Chen et al.'s confusion zones
5. A compute-efficiency analysis (K=8 vs K=64) for practical deployment

### Key Differentiators from Related Papers
| Feature | BPFC (ours) | Energy of Falsehood | TDGNet | CZEC (original) |
|---|---|---|---|---|
| Theoretical grounding | ✅ Doyle 2025 | ❌ | ❌ | ❌ |
| Internal (no extra components) | ✅ | ❌ NLI critic | ✅ | ✅ |
| During generation | ✅ | ❌ post-hoc | ❌ post-hoc | ✅ |
| Factual QA focus | ✅ | ✅ | ✅ | ✅ |
| Knowledge boundaries | ✅ | ❌ | ❌ | ✅ |
| AR comparison | ✅ | ❌ | ❌ | ✅ |
| Compute-efficiency analysis | ✅ | ❌ | ❌ | ❌ |

---

## #2 — Denoising-Time Self-Verification as Meta-Cognition

**Novelty score: 8/10** | **Feasibility: 8.5/10** | **Impact: 8/10**

DLMs can revise already-unmasked tokens (unique ability — AR models cannot). This creates a
"revision loop" that constitutes a richer form of meta-cognition.

Prism (Feb 2026) uses self-verification for hierarchical search decoding, but never studies
it as a meta-cognitive phenomenon or compares to AR's inability to revise.

**Experiment**: Mid-trajectory intervention study — inject factual errors at t_mid, measure:
- Self-correction rate as a function of: error salience, step position, entity frequency
- Whether correction depends on posterior variance (high σ² positions → more likely corrected)

---

## #3 — Knowledge Boundary Circuits via Posterior Variance + SAE Probing

**Novelty score: 9/10** | **Feasibility: 7/10** | **Impact: 9/10**

**New synthesis** (from Run #2): Use per-position posterior variance σ²_i as a mechanistic
probe — positions with high variance are drawing on stored knowledge. Connect to DLM-Scope
SAE features at those positions.

This is the DLM equivalent of ROME/MEMIT for AR models but uses posterior variance
(not causal intervention) to localize factual storage.

**Why novel**: ROME needs causal traces (left-to-right). DLMs are bidirectional — posterior
variance provides an alternative localization signal that exploits global context.

**Experiment**: For each (entity, attribute) factual pair:
1. Compute σ²_i across all positions during factual recall
2. Identify peak-σ² positions → are these the "attribute" positions?
3. Probe DLM-Scope SAEs at peak-σ² positions for factual vs. counterfactual inputs
4. Hypothesis: peak-σ² positions activate specific "factual query" SAE features

---

## #4 — Epistemic Decoding Order: Last-Unmasked = Least Certain

**Novelty score: 7.5/10** | **Feasibility: 8/10** | **Impact: 7.5/10**

The decoding ORDER in masked diffusion LMs (which position unmasks first) is a learned
strategy. MAGE (Feb 2026) shows all-mask attention encodes "where to look" proto-planning.

**Hypothesis**: Answer tokens for KNOWN facts unmask early and cleanly. Answer tokens for
HALLUCINATED outputs unmask late and hesitantly (multiple mask/unmask cycles).

**Simple experiment**: Track unmasking order + flip frequency per position on TriviaQA.
Correlate late-resolution / high-flip positions with factual incorrectness.

---

## #5 — Bidirectional vs Causal Uncertainty Architecture Comparison

**Novelty score: 7/10** | **Feasibility: 9/10** | **Impact: 8/10**

Side-by-side calibration curve comparison: LLaDA-8B vs LLaMA-8B on TriviaQA.
Semantic entropy (5 samples), verbalized confidence, posterior variance, ECE.

This is the clearest "macro" paper establishing that DLMs have different (and potentially
better) uncertainty representations than AR models.

---

## Removed / Deprioritized

| Direction | Status | Reason |
|---|---|---|
| Pure CZEC (entropy-only confusion zones) | Merged into #1 | Posterior variance is more rigorous |
| Meta-Learning Transfer (MAML for DLMs) | Dropped | Off-topic, low feasibility |
| Planning vs Reactive Generation | Low priority | Covered by MAGE |

---

## Immediate Actions

- [ ] Read Doyle (2507.07586) full PDF — understand exact assumptions for the Bayesian posterior proof
- [ ] Read Energy of Falsehood paper PDF — confirm our differentiation is accurate
- [ ] Set up LLaDA-8B inference with K-pass trajectory extraction
- [ ] Pilot: 100 TriviaQA questions, K=8, measure σ²_span vs correctness (AUROC)
- [ ] Write differentiation memo: BPFC vs Energy of Falsehood vs TDGNet
- [ ] Begin literature review section of draft paper
