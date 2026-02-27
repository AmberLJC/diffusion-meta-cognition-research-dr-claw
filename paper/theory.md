# Section 3: Theoretical Foundations of BPFC

## 3.1 Masked Discrete Diffusion Language Models

We work with **Masked Diffusion Language Models (MDLMs)**, specifically the LLaDA family (Lin et al., 2025), which defines a forward-reverse Markov process on discrete token sequences.

**Forward process.** Given a clean token sequence $\mathbf{x}_0 \in \mathcal{V}^L$ (vocabulary $\mathcal{V}$, length $L$), the forward process independently masks each token with probability $\alpha(t)$ at noise level $t \in [0, 1]$:

$$q(x_t^i \mid x_0^i) = (1 - \alpha(t)) \cdot \delta_{x_0^i} + \alpha(t) \cdot \delta_{\texttt{[MASK]}}$$

At $t=1$, all tokens are masked: $\mathbf{x}_1 = [\texttt{MASK}]^L$. The sequence is fully corrupted.

**Reverse process.** LLaDA learns a denoising network $p_\theta(\mathbf{x}_0 \mid \mathbf{x}_t)$ that approximates the reverse:

$$p_\theta(\mathbf{x}_0 \mid \mathbf{x}_t) = \prod_{i=1}^{L} p_\theta(x_0^i \mid \mathbf{x}_t)$$

where each token is predicted independently conditioned on the full noisy context. During generation, LLaDA performs $T$ denoising steps with **low-confidence remasking**: at each step, low-confidence tokens are randomly remasked and re-predicted, encouraging global consistency.

**Absorbing state structure.** The mask token $\texttt{[MASK]}$ is an absorbing state: once a token is unmasked (revealed) with sufficiently high confidence, it remains fixed. This gives MDLMs a distinctly non-AR generation dynamic: all positions are simultaneously refined.

---

## 3.2 Bayesian Posterior Interpretation (Doyle, 2025)

The central theoretical contribution of Doyle (arXiv:2507.07586) establishes that LLaDA's denoising process implements *exact* Bayesian posterior inference under mild assumptions.

**Theorem 3.1 (Doyle, 2025, Theorem 2).** *Let $\mathbf{x}_t$ be a noisy observation of $\mathbf{x}_0$ under the absorbing MDLM forward process. Then the optimal denoising distribution $p_\theta^*(\mathbf{x}_0 \mid \mathbf{x}_t)$ equals the exact Bayesian posterior:*

$$p_\theta^*(\mathbf{x}_0 \mid \mathbf{x}_t) = \frac{p(\mathbf{x}_t \mid \mathbf{x}_0) \cdot p(\mathbf{x}_0)}{p(\mathbf{x}_t)}$$

*where $p(\mathbf{x}_0)$ is the empirical distribution of the training corpus.*

**Implication.** A perfectly trained MDLM does not generate a single answer — it samples from the Bayesian posterior over all possible completions, weighted by training data evidence. When the model "knows" a fact, the posterior is sharply peaked at the correct answer. When the model is uncertain, the posterior is diffuse across multiple plausible completions.

**Corollary 3.2 (K-sample Monte Carlo consistency).** *For $K$ independent samples $\{\mathbf{y}^{(k)}\}_{k=1}^K$ from $p_\theta(\cdot \mid \mathbf{x}_t)$, the empirical variance converges to the true posterior variance at rate $O(1/\sqrt{K})$:*

$$\widehat{\sigma}^2_K(\mathbf{y}) := \frac{1}{K-1} \sum_{k=1}^K d(\mathbf{y}^{(k)}, \bar{\mathbf{y}})^2 \xrightarrow{K \to \infty} \text{Var}_{p_\theta}[\mathbf{y}]$$

*where $d$ is any consistent discrepancy measure and $\bar{\mathbf{y}}$ is the mean or mode of the $K$ samples.*

This corollary justifies our core approach: **K independent denoising passes provide a Monte Carlo estimate of the posterior variance**, which we use as an epistemic uncertainty signal.

---

## 3.3 The σ²_span Signal: Two Modes

We define two operationalizations of posterior variance for factual QA, corresponding to the granularity of available output signals.

### 3.3.1 Mode A: Answer-Level Variance (σ²_answer)

Let $Q$ be a factual question and $\{a^{(1)}, \ldots, a^{(K)}\}$ be $K$ independently sampled full answers from LLaDA. Define pairwise lexical agreement:

$$\text{agree}(a^{(j)}, a^{(k)}) = \mathbb{1}[\text{normalize}(a^{(j)}) = \text{normalize}(a^{(k)})]$$

where normalization strips punctuation, casing, and common articles (following TriviaQA evaluation protocol). Then:

$$\sigma^2_{\text{answer}} = 1 - \frac{2}{K(K-1)} \sum_{j < k} \text{agree}(a^{(j)}, a^{(k)})$$

$\sigma^2_{\text{answer}} \in [0, 1]$, where 0 means all K answers are identical (maximum confidence) and 1 means all K answers differ (maximum uncertainty).

**Properties:**
- Computationally trivial from API outputs
- Coarse-grained: treats answer as atomic
- Sensitive to paraphrase artifacts (two answers saying the same thing in different words inflate variance)
- Robust baseline compatible with any black-box text API

### 3.3.2 Mode B: Token-Level Variance (σ²_span) — Main Contribution

LLaDA's DenoiseViz output exposes **per-token confidence scores** $c_i^{(k)} \in [0, 1]$ for each token position $i$ in denoising pass $k$. These scores derive from LLaDA's internal softmax outputs during the final low-confidence remasking step:

$$c_i^{(k)} = p_\theta(x_0^i = \hat{x}_i^{(k)} \mid \mathbf{x}_t^{(k)})$$

where $\hat{x}_i^{(k)}$ is the predicted token at position $i$ in pass $k$.

Given $K$ passes, we define the **token-level posterior variance** for position $i$:

$$\sigma^2_i = \text{Var}_k[c_i^{(k)}] = \frac{1}{K-1} \sum_{k=1}^K (c_i^{(k)} - \bar{c}_i)^2, \quad \bar{c}_i = \frac{1}{K}\sum_k c_i^{(k)}$$

The **span variance** $\sigma^2_{\text{span}}$ averages over the answer-token positions $\mathcal{A} = \{i : \text{position } i \text{ is in the answer span}\}$:

$$\sigma^2_{\text{span}} = \frac{1}{|\mathcal{A}|} \sum_{i \in \mathcal{A}} \sigma^2_i$$

**Theoretical connection.** By Doyle's Theorem 3.1, $c_i^{(k)}$ is the model's Bayesian posterior probability over token $x_0^i$ at position $i$. High $\sigma^2_i$ means the model's posterior probability over the correct token oscillates across passes — the hallmark of epistemic uncertainty. This is precisely the "low-confidence remasking" signal that drives oscillation in "confusion zones" (DLM-Scope, arXiv:2511.15208).

**Why Mode B is theoretically superior to Mode A:**
1. Mode A discards the confidence structure — two answers can be identical tokens with very different internal certainties
2. Mode B captures "uncertain but consistent" behavior: K passes all output the same token but with low $c_i^{(k)}$ — the model is *guessing* consistently, not *knowing*
3. Mode B separates epistemic uncertainty (high $\sigma^2_i$) from aleatoric ambiguity (consistently low $\bar{c}_i$)
4. Mode B provides position-specific diagnostics — which part of the answer is uncertain?

---

## 3.4 Calibration: BPFC as a Proper Score

We claim that $\sigma^2_{\text{span}}$ (or $\sigma^2_{\text{answer}}$) constitutes a **calibrated epistemic uncertainty measure** for factual QA.

**Definition 3.3 (Calibration).** An uncertainty measure $u(Q)$ is *calibrated* if the model's empirical accuracy on questions with uncertainty score $u$ equals the predicted confidence $1 - u$:

$$\mathbb{E}[\mathbf{1}[\text{correct}(Q)] \mid u(Q) = s] = 1 - s$$

We do not claim perfect calibration (which would require a perfectly trained model). Instead, we claim the weaker **calibration monotonicity** property: $u(Q)$ is positively rank-correlated with prediction error:

$$\text{rank-corr}(u(Q), \mathbf{1}[\text{incorrect}(Q)]) > 0$$

This is measured via **AUROC**: the probability that a random incorrect answer has higher $\sigma^2_{\text{span}}$ than a random correct answer. We expect AUROC > 0.5, with AUROC → 1 under perfect Bayesian calibration.

**Expected Calibration Error (ECE)** provides a quantitative calibration measure:

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left| \text{acc}(B_b) - (1 - \bar{u}(B_b)) \right|$$

where $B_b$ are equal-frequency bins of questions sorted by $u(Q)$.

---

## 3.5 BPFC vs. Semantic Entropy (Kuhn et al., 2023)

Semantic Entropy (SE) for AR models uses K temperature samples to compute entropy over semantic equivalence classes. BPFC differs in three key ways:

| Property | BPFC (DLM) | Semantic Entropy (AR) |
|---|---|---|
| **Sampling mechanism** | K independent denoising chains from $\mathbf{x}_1 = [\texttt{MASK}]^L$ | K temperature samples from $p_\theta(y_t \mid y_{<t})$ |
| **Theoretical grounding** | Doyle (2025): exact Bayesian posterior | Approximate posterior via temperature annealing |
| **Granularity** | Per-token $\sigma^2_i$ (Mode B) | Sequence-level entropy |
| **Token temperature** | No temperature parameter; stochasticity from masking | Requires temperature tuning (too high → nonsense, too low → degenerate) |
| **Black-box compatible** | Yes (DenoiseViz output) | Yes (text output only, no logits needed) |

The absence of a temperature hyperparameter in BPFC is a practical advantage: AR-SE requires tuning temperature per model and domain, whereas BPFC's stochasticity is intrinsic to the denoising process and theoretically grounded.

---

## 3.6 Connection to Knowledge Boundary Estimation

Mallen et al. (2023, PopQA) showed that entity popularity $f(e)$ (Wikipedia view frequency) predicts AR accuracy via:

$$P(\text{correct}(Q) \mid e) \approx \sigma\left(\beta_0 + \beta_1 \log f(e)\right)$$

We extend this with the following conjecture, which the BPFC pilot will test:

**Conjecture 3.4 (BPFC-Knowledge Boundary).** *For an MDLM trained on a corpus $\mathcal{C}$, the expected σ²_span satisfies:*

$$\mathbb{E}[\sigma^2_{\text{span}} \mid e] \approx g\left(\frac{1}{f_\mathcal{C}(e)}\right)$$

*where $f_\mathcal{C}(e)$ is the training corpus frequency of entity $e$ and $g$ is a monotonically increasing function.*

**Intuition**: Frequently-occurring entities appear in many contexts during training, sharpening the posterior $p_\theta(\cdot \mid Q)$ and reducing $\sigma^2_{\text{span}}$. Rare entities produce diffuse posteriors (high $\sigma^2_{\text{span}}$) even when the model occasionally produces the correct answer by chance ("lucky guess"). This decomposition — *genuinely known* (low $\sigma^2_{\text{span}}$, high accuracy) vs. *lucky guess* (high $\sigma^2_{\text{span}}$, high accuracy) vs. *genuinely unknown* (high $\sigma^2_{\text{span}}$, low accuracy) — provides a richer characterization of LLM knowledge boundaries than accuracy alone.

---

## 3.7 Summary of Theoretical Claims

| Claim | Basis | Testable? |
|-------|-------|-----------|
| K passes → MC estimate of posterior variance | Doyle (2025) Cor. 3.2 | Yes (convergence as K increases) |
| Token-level $c_i^{(k)}$ = Bayesian posterior prob | Doyle (2025) Thm 3.1 + DenoiseViz | Yes (correlation with accuracy) |
| σ²_span positively rank-corr with error | Calibration monotonicity (Def 3.3) | Yes (AUROC > 0.5) |
| High σ²_span ↔ rare entity / knowledge boundary | Conjecture 3.4 | Yes (entity-frequency correlation) |
| BPFC outperforms verbalized confidence | Structural vs. self-report | Yes (compare to "how confident are you?") |

These five claims constitute the empirical program of BPFC. The pilot experiment (Section 4) tests claims 3 and 4 as the primary targets, with claims 1 and 2 as secondary analyses.

---

*[Section written by Dr. Claw, 2026-02-27]*
