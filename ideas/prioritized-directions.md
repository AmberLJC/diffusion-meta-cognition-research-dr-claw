# Prioritized Research Directions
_Last updated: 2026-02-27 00:11 UTC_

---

## 🏆 #1 — Confusion-Zone Epistemic Calibration (CZEC)

**Novelty score: 9.5/10** | **Feasibility: 8/10** | **Impact: 9/10**

### The Claim
The geometry of "confusion zones" (entropy spikes during diffusion LLM denoising) encodes epistemic uncertainty and can serve as a zero-shot calibration signal — i.e., a proxy for whether the model *knows* the answer or is hallucinating.

### Why Novel
- Confusion zones were discovered by Chen et al. (Nov 2025) but only studied in the context of RL training efficiency (ATPO)
- Nobody has asked: does confusion zone geometry correlate with **factual knowledge**?
- TDGNet uses temporal attention for hallucination detection — different signal, different method
- DLM-Scope provides SAE features that could be probed at confusion zone timesteps — unexplored intersection
- All AR calibration methods (semantic entropy, P(IK), conformal prediction) have zero DLM equivalents

### Formal Research Question
> Can we extract a calibrated epistemic uncertainty signal from the denoising trajectory of a masked diffusion LM, using confusion zone features (entropy curve shape, RoEC peaks, CM trajectory), without any additional training?

### Proposed Experiments
1. **Pilot (1 week)**: Run LLaDA-8B on TriviaQA; extract trajectory confusion features; correlate with correctness (AUROC)
2. **Knowledge stratification (2 weeks)**: Segment by entity popularity; test hypothesis that "unknown" facts → higher confusion mass
3. **SAE + Confusion joint signal (3 weeks)**: Use DLM-Scope SAEs at confusion zone steps to identify knowledge boundary circuits

### Key Baselines
- Token-level confidence (Search-or-Accelerate style)
- TDGNet (attention-graph hallucination detection)
- AR semantic entropy (Kuhn et al.)
- Verbalized uncertainty from LLaDA vs GPT-4o

### Expected Contribution
A new calibration paradigm for DLMs that exploits temporal trajectory dynamics — fundamentally impossible for AR models. First paper connecting confusion zones to epistemics.

---

## #2 — Denoising-Time Self-Verification as Meta-Cognition

**Novelty score: 8/10** | **Feasibility: 8.5/10** | **Impact: 8/10**

### The Claim
Diffusion LMs can be made to "self-verify" their own factuality mid-trajectory, not just at the end (as in Prism). This mid-trajectory self-verification is a richer form of meta-cognition than AR equivalents because it allows *revision* of already-placed tokens.

### Why Novel
- Prism (Feb 2026) uses self-verification for decoding search — not studied as a meta-cognitive phenomenon
- AR self-verification papers exist (self-consistency, etc.) but can't revise past tokens
- DLMs' ability to un-mask tokens gives a unique "revision loop" — completely unexplored for meta-cognition

### Proposed Experiments
1. **Intervention study**: At step t_mid, inject factual errors; does the DLM self-correct?
2. **Measure self-correction rates** as a function of: (a) error salience, (b) step position, (c) entity frequency
3. **Compare to AR**: GPT can't revise committed tokens — DLMs may have fundamentally superior correction ability

---

## #3 — Knowledge Boundary Circuits in DLMs via Probing

**Novelty score: 8.5/10** | **Feasibility: 7/10** | **Impact: 8.5/10**

### The Claim
There exist specific circuits (attention head patterns or SAE feature activations) in DLMs that reliably activate when the model encounters knowledge boundaries — i.e., regions where factual recall fails. These circuits are detectable and different from what AR LLMs develop.

### Why Novel
- DLM-Scope (Feb 2026) found that SAE features are interpretable and stable in DLMs
- DLM-Scope shows SAEs useful for decoding order — nobody has extended to knowledge/factuality
- AR mechanistic interpretability has studied "knowledge storage" in MLP layers (Meng et al., ROME) — DLM equivalent is completely unexplored
- In DLMs, knowledge queries happen in bidirectional context, not causal — storage circuits may be fundamentally different

### Proposed Experiments
1. **Probe DLM-Scope SAE features** for factual vs counterfactual inputs
2. **Circuit tracing**: Which attention heads activate on "entity → attribute" queries in DLMs?
3. **Compare**: AR ROME-style localization vs DLM-style — where is knowledge stored?

---

## #4 — Epistemic Tokens in Masked Diffusion: Emergence of "Uncertainty Flags"

**Novelty score: 7.5/10** | **Feasibility: 7/10** | **Impact: 7.5/10**

### The Claim
In masked diffusion LMs, certain token positions may develop a functional role as "epistemic anchors" — positions that remain uncertain (masked or low-confidence) longer than others and whose resolution sequence encodes the model's implicit confidence ordering.

### Why Novel
- MAGE (Feb 2026) shows that all-mask attention already "knows where to look" — but doesn't ask *why* certain positions resist early unmasking
- The "decoding order" in DLMs is a learned strategy — could encode implicit confidence
- No paper has studied whether decoding order correlates with epistemic uncertainty

### Proposed Experiments
1. **Track unmasking order** for factually known vs unknown questions
2. **Hypothesis**: Answer tokens unmask later (and in more confused manner) for hallucinated answers
3. **Correlate**: DLM-Scope decoding order signals with factual accuracy

---

## #5 — Bidirectional Uncertainty vs Causal Uncertainty: Direct Architecture Comparison

**Novelty score: 7/10** | **Feasibility: 9/10** | **Impact: 8/10**

### The Claim
Diffusion LMs, by virtue of bidirectional context from step 0, develop fundamentally different uncertainty representations than AR models. This is measurable via probing tasks, calibration curves, and semantic entropy comparisons.

### Proposed Experiments
1. **Side-by-side calibration curves**: LLaDA-8B vs LLaMA-8B on TriviaQA
2. **Semantic entropy** (Kuhn-style): sample 10 answers from each, cluster, compare calibration
3. **Verbalized confidence**: Do diffusion models express calibrated verbal uncertainty?

---

## Previously Considered Directions (Downgraded)

| Direction | Status | Reason Downgraded |
|---|---|---|
| Meta-Learning Transfer (MAML for DLMs) | Low priority | Hard, little infrastructure, indirect relevance |
| Planning vs Reactive Generation | Moderate | Now partially covered by MAGE and block diffusion papers |

---

## Next Research Steps

- [ ] Get LLaDA-8B weights and run pilot confusion-zone extraction on TriviaQA (100 questions)
- [ ] Read DLM-Scope paper PDF in detail for SAE architecture specifics
- [ ] Read TDGNet paper PDF for implementation details
- [ ] Set up trajectory logging code for LLaDA inference
- [ ] Write code to extract: entropy per step, RoEC, CM per position
