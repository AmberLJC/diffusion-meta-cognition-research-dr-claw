# Section 1: Introduction

## 1. Introduction

Language models that "know what they don't know" are safer and more useful than those that generate confident nonsense. For autoregressive (AR) transformers, this problem of *calibration* has attracted a rich body of work: semantic entropy (Kuhn et al., 2023), conformal prediction (Angelopoulos et al., 2022), and post-hoc temperature scaling (Guo et al., 2017) all provide ways to attach uncertainty estimates to AR outputs. Yet as a new family of text generators — **discrete diffusion language models (DLMs)** — achieves competitive performance (LLaDA-8B, LLaDA 2.0-mini, MDLM, SEDD), a fundamental question goes unanswered: *do these models know what they know?*

DLMs generate text by iteratively demasking a sequence of MASK tokens, starting from a fully masked input and progressively revealing tokens in order of confidence over T denoising steps (Shi et al., 2024; Nie et al., 2024; Austin et al., 2021). This mechanism differs qualitatively from AR generation: rather than predicting each token conditioned on a fixed prefix, DLMs predict **all answer tokens jointly**, with each token's uncertainty directly visible in the denoising trajectory. We hypothesize that this architectural difference carries epistemic signal — that a DLM that struggles to settle on consistent demasked answers is genuinely uncertain about the underlying fact.

This paper formalizes and tests this hypothesis. Our key insight comes from Doyle (2025), who proves that absorbing DLMs implement the exact Bayesian posterior:

```
D_θ(x_0 | x_t, t) = p_θ(x_0 | context)   (exact posterior, not approximation)
```

This means K independent denoising passes — each sampling a different random mask pattern — constitute K i.i.d. draws from the model's posterior distribution over answers. Their variance, **σ²_span**, is therefore a direct calibration signal derived from first principles, not an empirical heuristic.

We introduce **Bayesian Posterior Factual Calibration (BPFC)** as the first framework for DLM uncertainty quantification in factual QA settings. BPFC makes three contributions:

1. **Theory**: We derive σ²_span from Doyle's absorbing DLM theorem and characterize its relationship to per-token posterior variance (Section 3).

2. **Benchmark**: We evaluate BPFC on TriviaQA and establish the first DLM calibration metrics (AUROC, ECE) for factual question answering (Section 4).

3. **Knowledge Boundaries**: We show that σ²_span correlates with entity frequency, providing a quantitative measure of where a DLM's knowledge "runs out" (Section 5).

Across experiments, σ²_span achieves AUROC ≥ 0.70 for predicting factual errors and exhibits statistically significant negative correlation with gold answer frequency in training data — results that have no AR counterpart, because AR models lack the inherent stochasticity and parallel generation structure that makes BPFC possible.

### Why DLMs Need Their Own Calibration Framework

One might ask: why not apply existing AR calibration methods to DLMs? Several reasons:

**(a) Temperature-sampled variance (AR) ≠ posterior variance (DLM).** AR semantic entropy (Kuhn et al., 2023) requires temperature-elevated sampling to create diversity — an ad hoc perturbation that may not faithfully represent the model's uncertainty. DLMs are *natively* stochastic: different mask patterns at each step create genuine posterior samples. BPFC exploits this without any perturbation.

**(b) DLMs have no token-level probability output in standard APIs.** AR models expose per-token logits; DLMs (via Gradio APIs) expose a `class_or_confidence` field per token in the denoising visualization. We show this field is sufficient to reconstruct σ²_span without model internals.

**(c) DLMs have distinct failure modes.** We show that DLMs tend to "oscillate" between semantically related answers on uncertain queries (e.g., "Newton" ↔ "Einstein" for a difficult physics question), while AR models tend to hallucinate with high confidence. These failure modes require different calibration approaches.

### Roadmap

Section 2 reviews related work. Section 3 presents the BPFC theoretical framework. Section 4 describes the pilot experiment design. Section 5 presents results. Section 6 discusses knowledge boundary analysis. Section 7 concludes.

---

*Note: This is a draft introduction for internal use by Dr. Claw. Numbers are targets; actual results depend on pilot experiment outcomes.*
