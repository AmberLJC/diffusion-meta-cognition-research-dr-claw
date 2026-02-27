# Section 2: Related Work

## 2. Related Work

### 2.1 Discrete Diffusion Language Models

The modern discrete diffusion paradigm builds on the masked diffusion process introduced by Austin et al. (2021) and D3PM. LLaDA (Nie et al., 2024; arXiv:2502.09992) scales masked diffusion to 8B parameters with instruction tuning, demonstrating competitive performance with GPT-3.5-level AR models on reasoning benchmarks. LLaDA 2.0-mini (inclusionAI, 2025) extends this to 16B parameters with a Mixture-of-Experts design (1.4B active), achieving MMLU 80.53 and HumanEval 86.59 — state-of-the-art for DLMs. MDLM (Sahoo et al., 2024) and SEDD (Lou et al., 2024) provide theoretical alternatives to the absorbing noise schedule, while MD4 (Shi et al., 2024) further connects masked diffusion to language modeling objectives.

Our work is the first to study the *epistemic properties* of these models rather than their generative quality.

### 2.2 The Bayesian Posterior Result

**Doyle (2025)** [arXiv:2507.07586] is our primary theoretical foundation. Doyle proves that absorbing discrete diffusion language models implement the exact Bayesian posterior under mild regularity conditions: the denoiser D_θ(x_0 | x_t, t) approximates p_θ(x_0 | x_t) at each step. Monte Carlo estimates via K independent passes converge at rate O(1/√K), with empirical Spearman ρ = 0.996 between σ² and reconstruction error on WikiText-2. We are the first to apply this result to factual calibration in QA settings.

### 2.3 Calibration and Uncertainty in Autoregressive LLMs

**Semantic Entropy** (Kuhn et al., 2023; NeurIPS) clusters K temperature-sampled AR outputs by semantic equivalence (via NLI) and uses entropy of the resulting distribution as an uncertainty signal. Semantic entropy achieves AUROC ~0.73 on TriviaQA for GPT-3.5. BPFC is inspired by this paradigm but adapts it to DLMs, replacing ad hoc temperature-sampling with principled posterior sampling and NLI-based semantic clustering with lexical agreement (pilot) or embedding similarity (full study).

**Conformal Prediction** (Angelopoulos et al., 2022; Quach et al., 2023) provides distribution-free coverage guarantees for LLM outputs. BPFC is complementary: we provide a calibration *signal*, not a coverage *guarantee*. Combining BPFC with conformal prediction is a natural future direction.

**Temperature Scaling** (Guo et al., 2017) and post-hoc calibration methods assume access to model logits. DLMs do not expose logits in standard APIs; BPFC works from behavioral outputs alone.

**Verbalized Confidence** (Xiong et al., 2023; Lin et al., 2022) elicits self-reported uncertainty ("I'm 80% confident..."). DLMs can generate such text, but we argue σ²_span provides a *structural* signal independent of any verbalization capability.

### 2.4 Diffusion Models and Hallucination/Uncertainty

**The Energy of Falsehood** [arXiv:2507.10831] (2025) analyzes the energy function of continuous diffusion models (image/text) as a detector for hallucinated content. Key difference from BPFC: (a) focuses on continuous diffusion (SDEs), not discrete masked diffusion; (b) uses energy-based anomaly detection, not posterior variance; (c) targets hallucination *detection* post-hoc, not calibration of knowledge boundaries. Complementary direction.

**DLM-Scope** [arXiv:2511.15208] (Nov 2025) identifies "confusion zones" in LLaDA denoising trajectories where tokens oscillate between alternatives. This provides a step-level signal (which denoising steps are uncertain) whereas BPFC provides a pass-level signal (which questions are uncertain). The confusion zone phenomenon may explain why our σ²_span works: questions with high σ²_span should exhibit more confusion zones in their denoising trajectories.

**arXiv:2602.08920** (Dao et al., Feb 2026) retrofits *AR* transformers with diffusion-inspired uncertainty propagation for calibration. Fundamentally different: they modify AR architecture, we study native DLM behavior. We study the epistemics of actual diffusion inference; they use diffusion as an architectural metaphor for uncertainty.

### 2.5 Knowledge Boundary Estimation

**KNOW** (Amayuelas et al., 2023) and related work studies "what LLMs know" by testing accuracy as a function of entity frequency. We extend this to DLMs and show that σ²_span provides a finer-grained signal than accuracy alone: σ²_span discriminates "uncertain but lucky" (correct by accident, high variance) from "genuinely known" (correct with low variance).

**PopQA** (Mallen et al., 2023) establishes that entity popularity (Wikipedia page views) strongly predicts AR accuracy. We use similar entity-frequency stratification to analyze σ²_span, providing the first such analysis for DLMs.

### 2.6 What BPFC Does Not Do

For clarity, we distinguish BPFC from:
- **Discrete Stochastic Localization** [arXiv:2602.16169] (Feb 2026): training technique to improve MDLM step efficiency; no uncertainty/calibration component.
- **TDGNet / DLM-based fact verification**: These use DLMs as generative tools for fact-checking; BPFC studies DLMs' own uncertainty about facts.
- **Model-based conformal prediction**: We don't assume access to model internals or training data statistics.

---

*[TODO: Add Kadavath et al. (2022) "Language Models (Mostly) Know What They Know" as related work on self-knowledge.]*
*[TODO: Check if there are any DLM papers from ICLR 2026 (currently in review) relevant to calibration.]*
