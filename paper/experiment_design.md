# Section 4: Experiment Design

## 4.1 Overview

We conduct a **pilot study** (N=50 questions, K=8 passes) to establish feasibility and obtain preliminary AUROC/ECE estimates, followed by a **full evaluation** (N=200 questions, K=16 passes) for publication. This section describes the complete experimental design.

### Primary Research Questions

- **RQ1**: Does σ²_answer (Mode A, black-box) correlate with prediction error on TriviaQA? (AUROC > 0.5?)
- **RQ2**: Does σ²_span (Mode B, token-level) provide a stronger calibration signal than σ²_answer?
- **RQ3**: Does σ²_span correlate with entity frequency, consistent with Conjecture 3.4?
- **RQ4**: How does BPFC compare to Semantic Entropy on the same questions for GPT-4o-mini?

---

## 4.2 Dataset

**Primary dataset**: TriviaQA (Joshi et al., 2017), specifically the **unfiltered Web validation set**.

**Rationale**: TriviaQA provides:
1. Gold answers with multiple valid normalizations (handles paraphrase in σ²_answer)
2. Entity annotations enabling RQ3 (frequency analysis)
3. Well-studied AR baselines (Kuhn et al. 2023 used similar benchmarks)
4. Wide difficulty range — from very common (low σ²_span expected) to obscure facts (high σ²_span expected)

**Sampling protocol** (to avoid biases):
```python
# Stratified by estimated difficulty
bins = {"easy": [], "medium": [], "hard": []}
for q in triviaqa_dev:
    freq = get_entity_frequency(q)  # Wikipedia pageviews via API
    if freq > 1e6:    bins["easy"].append(q)
    elif freq > 1e4:  bins["medium"].append(q)
    else:             bins["hard"].append(q)

# Sample N//3 from each difficulty bin
sample = (
    random.sample(bins["easy"],   N // 3) +
    random.sample(bins["medium"], N // 3) +
    random.sample(bins["hard"],   N - 2 * (N // 3))
)
```

For the pilot (N=50): ~17 easy, 17 medium, 16 hard.
For the full evaluation (N=200): 67 easy, 67 medium, 66 hard.

**Preprocessing**:
- Filter questions where gold answer length > 5 tokens (to avoid σ²_span boundary ambiguity)
- Exclude questions requiring arithmetic (to avoid conflating procedural and factual uncertainty)
- All questions formatted as: `"Answer the following question in one or two words: {question}"`

---

## 4.3 Model: LLaDA-8B-Instruct

**Access method**: HuggingFace Space `multimodalart/LLaDA` via Gradio v5 API (ZeroGPU, free compute).

**Why LLaDA-8B-Instruct**:
- Only publicly accessible instruction-following MDLM at scale (8B params)
- ZeroGPU grant provides free A100 inference (eliminating cost barrier)
- DenoiseViz output available (enables Mode B, σ²_span)
- Established TriviaQA baseline does not yet exist (first-mover advantage)

**Model parameters** (as configured in the Space):
- `gen_length`: 128 (answer generation window)
- `steps`: 128 (denoising steps; LLaDA default)
- `block_length`: 32 (semi-AR blocks)
- Temperature: not user-adjustable (stochasticity from masking only)

**K independent passes**: Each "pass" is a fresh API call with the same prompt. Independence is guaranteed because:
1. ZeroGPU uses stateless workers per request
2. The forward process starts fresh from $\mathbf{x}_1 = [\texttt{MASK}]^L$ each time
3. Any session state (chat history) is cleared between passes

**Practical note on stateful Gradio API**: The LLaDA Space uses `gr.State` for chat history. Fresh API calls may initialize this state as `None` rather than `[]`. Workaround: use `gradio_client` Python library (which handles stateful lifecycle) or manually include the initialization call:
```python
# Step 1: Initialize session with empty state
client.predict(message="", history=[], api_name="/user_message_submitted")
# Step 2: Generate response (bot_response endpoint)
result = client.predict(history=[], api_name="/bot_response")
```

---

## 4.4 BPFC Measurement Protocol

### Mode A: σ²_answer (Answer-Level Variance)

```
For each question Q:
  answers = []
  For k in 1..K:
    a_k = call_llada_api(prompt=format_prompt(Q))
    a_k_norm = normalize_answer(a_k)   # lowercase, strip punct/articles
    answers.append(a_k_norm)
  
  # Compute pairwise agreement
  agree_count = sum(a_j == a_k for j < k) 
  sigma2_answer = 1 - (2 * agree_count) / (K * (K-1))
  
  # Correctness: does any answer match gold?
  correct = any(gold_match(a, gold_answers) for a in answers)
```

**Normalization** follows TriviaQA evaluation script:
- Lowercase
- Remove articles: "a", "an", "the"
- Remove punctuation (keep alphanumeric and spaces)
- Strip leading/trailing whitespace
- For multi-word answers: also check exact match after tokenization

### Mode B: σ²_span (Token-Level Variance)

```
For each question Q:
  token_confidences = []  # shape: K × L
  For k in 1..K:
    result = call_llada_api_with_denoiseviz(prompt=format_prompt(Q))
    confs_k = extract_token_confidences(result["denoiseviz"])
    # confs_k[i] = final denoising step confidence for position i
    token_confidences.append(confs_k)
  
  # Identify answer span positions (positions after [SEP] or answer trigger)
  answer_positions = identify_answer_span(token_confidences)
  
  # Compute per-token variance across K passes
  sigma2_i = np.var([token_confidences[k][i] for k in range(K)], ddof=1)
  
  # Average over answer span
  sigma2_span = np.mean([sigma2_i for i in answer_positions])
```

**DenoiseViz output format** (confirmed from Gradio v5 API schema):
```json
{
  "Denoising Process Visualization": [
    {"token": "Paris", "class_or_confidence": 0.94},
    {"token": "is", "class_or_confidence": 0.87},
    {"token": "the", "class_or_confidence": 0.91},
    ...
  ]
}
```

**Answer span identification**: We identify the answer span by finding the tokens following the prompt's question mark or "Answer:" trigger in the generated output. For robustness, we use the last 5-15 positions of the generated content as the answer span when explicit span boundaries are ambiguous.

---

## 4.5 AR Baseline: Semantic Entropy

To benchmark BPFC against the state of the art, we compute **Semantic Entropy** (Kuhn et al., 2023) for GPT-4o-mini on the same questions.

**Protocol**:
```python
# For each question Q:
samples = []
for k in range(K):  # K=8, same as BPFC
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": format_prompt(Q)}],
        temperature=0.7,  # Standard SE temperature
        max_tokens=50
    )
    samples.append(response.choices[0].message.content)

# Cluster by semantic equivalence (using NLI or string match)
clusters = cluster_semantic_equivalents(samples)

# Entropy over cluster distribution
p_c = [len(c)/K for c in clusters]
SE = -sum(p * log(p) for p in p_c if p > 0)
```

**Cost estimate**: 50 questions × 8 samples × ~100 tokens = 40K tokens ≈ **$0.006 total**.

**Additional AR comparisons**:
- **Verbalized confidence**: Ask GPT-4o-mini "How confident are you? (0-100%)" after each answer; compare to σ²_answer
- **Self-consistency** (Wang et al., 2022): Same K=8 samples; confidence = fraction agreeing with majority

---

## 4.6 Evaluation Metrics

### Primary: AUROC (Area Under ROC Curve)

$$\text{AUROC}(u, \text{err}) = P(u(Q_{\text{wrong}}) > u(Q_{\text{right}}))$$

- $u$ = uncertainty measure (σ²_answer, σ²_span, SE, etc.)
- Computed via `sklearn.metrics.roc_auc_score`
- **Target**: AUROC > 0.60 (strong signal); > 0.50 (any signal)
- **Null hypothesis**: AUROC = 0.50 (random)

### Secondary: Expected Calibration Error (ECE)

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} |\text{acc}(B_b) - (1 - \bar{u}(B_b))|$$

- $B=10$ equal-frequency bins sorted by $u$
- **Target**: ECE < 0.15 (well-calibrated)

### RQ3: Entity Frequency Correlation

$$\rho = \text{Pearson}(\sigma^2_{\text{span}}, -\log f(e))$$

- $f(e)$ = Wikipedia pageviews for the answer entity
- **Target**: $\rho > 0.30$ (moderate positive correlation)

### Knowledge Boundary Analysis

Four-quadrant decomposition (predicted from Conjecture 3.4):
```
                    Low σ²_span     High σ²_span
                ┌───────────────┬────────────────┐
  Correct       │  "Known"      │  "Lucky Guess" │
                │  (target: ~40%)│  (target: ~10%)│
                ├───────────────┼────────────────┤
  Incorrect     │  "Confident   │  "Unknown"     │
                │   Mistake"    │  (target: ~35%)│
                │  (target: ~15%)│               │
                └───────────────┴────────────────┘
```
Threshold: σ²_span median split. "Lucky Guess" quadrant demonstrates BPFC's advantage over accuracy alone.

---

## 4.7 Statistical Analysis

- **Sample size justification**: N=50 provides 80% power to detect AUROC=0.65 vs. 0.50 (two-sided, α=0.05) at $n=45$ correct/incorrect split.
- **Confidence intervals**: Bootstrap (B=1000) for AUROC, ECE, and Pearson ρ.
- **Multiple comparison correction**: Bonferroni for 4 primary metrics; report raw p-values as well.
- **Effect size**: Cohen's d for Mode A vs Mode B σ² comparison; Cliff's δ for rank-based comparisons.

---

## 4.8 Infrastructure and Runtime

### API Access

| Component | Service | Cost | Rate Limit |
|-----------|---------|------|------------|
| LLaDA (Mode A+B) | HF Space ZeroGPU | Free | ~1 req/30s |
| GPT-4o-mini (AR baseline) | OpenAI API | ~$0.009 | 3500 RPM |
| Entity frequency | Wikipedia API | Free | 200/s |
| TriviaQA data | HF Datasets | Free | — |

### Runtime Estimate

**Sequential upper bound**: 50 questions × 8 passes × 35s/pass = **3.9 hours**
**Parallelized (3 concurrent)**: ~**80 minutes**

The `bpfc_pilot.py` implementation uses `asyncio` with `asyncio.Semaphore(3)` to cap concurrent API calls.

### Experiment Code Location

- `experiments/bpfc_pilot.py` — Main experiment runner
- `data/triviaqa_sample.jsonl` — Sampled questions (auto-downloaded)
- `data/bpfc_pilot_results.jsonl` — Per-question results
- `data/bpfc_pilot_analysis.json` — Aggregate metrics
- `data/entity_frequencies.json` — Cached Wikipedia pageview data

---

## 4.9 Ablations (Full Paper Version)

For the full N=200 evaluation, we run the following ablations:

| Ablation | Hypothesis | What Varies |
|----------|------------|-------------|
| K sensitivity | AUROC improves with K up to K≈8 | K ∈ {1, 2, 4, 8, 16} |
| Prompt format | σ²_span robust to phrasing | 3 prompt variants |
| Semantic vs. lexical | Mode A with BERTScore vs. exact match | Agreement metric |
| Answer span length | σ²_span robust to answer length | Answer length ∈ {1, 2, 3+} tokens |
| Model: LLaDA 2.0-mini | Does a smaller/newer model show similar pattern? | Model variant |

---

## 4.10 Experimental Hypotheses (Preregistered)

To reduce HARKing risk, we state directional hypotheses before running:

**H1**: σ²_answer AUROC > 0.55 on TriviaQA at N=50, K=8.  
**H2**: σ²_span AUROC > σ²_answer AUROC (Mode B outperforms Mode A).  
**H3**: Pearson ρ(σ²_span, -log f(e)) > 0.25.  
**H4**: BPFC σ²_span AUROC within 0.05 of GPT-4o-mini Semantic Entropy AUROC (competitive despite no logit access).  

**Failure modes and interpretations**:
- If H1 fails (AUROC ≈ 0.50): LLaDA may not have sufficient factual knowledge / DenoiseViz may not reflect epistemic uncertainty. Pivot to LLaDA 2.0 or different benchmark.
- If H2 fails (Mode B ≤ Mode A): Token-level variance may be dominated by formatting noise. Investigate span identification quality.
- If H4 fails (BPFC AUROC << SE): Accept that DLMs have weaker epistemic calibration; this is itself a publishable finding.

---

*[Section written by Dr. Claw, 2026-02-27]*
