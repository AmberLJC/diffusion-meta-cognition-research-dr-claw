# 🖥️ GPU-Free Experiment Strategy

No GPU available. Every experiment must be designed around this constraint while still producing meaningful, publishable results.

## Allowed Resources
- **CPU compute** (this machine)
- **HuggingFace Inference API** (free tier, hosted models) — LLaDA, MDLM, etc.
- **Together AI / Replicate APIs** (pay-per-token, cheap for small pilots)
- **OpenAI / Anthropic APIs** — for autoregressive baselines
- **Pre-existing datasets** — TriviaQA, TruthfulQA, SQuAD (no generation needed, just scoring)
- **Cached/pre-computed model outputs** from papers (check HuggingFace datasets)
- **Small proxy models** — BERT-scale diffusion models runnable on CPU

## Experiment Design Principles

### 1. Use APIs, not local inference
- LLaDA-8B via HF Inference API or Together AI
- GPT-4o-mini as AR baseline (cheap, ~$0.001/query)
- Run pilots on N=50-100 samples (statistically meaningful, cheap)

### 2. Analyze existing artifacts
- Paper authors often release logits / trajectory data
- Mine arXiv supplementals and HuggingFace datasets for pre-computed embeddings
- Re-analyze existing benchmark results with new metrics

### 3. Theoretical contributions
- Mathematical analysis of confusion zone geometry doesn't need compute
- Proofs about denoising trajectory properties
- Taxonomy papers, conceptual frameworks

### 4. Lightweight proxy experiments
- Use tiny models (< 500M params) as CPU-feasible proxies
- Validate that patterns hold at small scale → argue they generalize

### 5. Prompt-based behavioral experiments
- Test meta-cognitive behaviors via prompting (no internal access needed)
- Compare uncertainty expressions, refusal rates, confidence language

## Current Experiment Plan (CZEC)

**Goal:** Test whether confusion zone metrics predict factual accuracy in LLaDA

**GPU-free version:**
1. Use HF Inference API to run LLaDA-8B on 50 TriviaQA questions (free tier)
2. Request full token probability outputs at each denoising step
3. Compute confusion_mass, RoEC_peak from returned logits (CPU numpy, trivial)
4. Compare to ground truth → AUROC
5. AR baseline: GPT-4o-mini logprobs on same 50 questions (~$0.01 total)

**Estimated cost:** < $1 total
**Estimated time:** 2-3 hours of API calls + analysis

## Backup: Purely Theoretical Paper
If API access is insufficient, pivot to a theoretical contribution:
- Formal definition of confusion zones in masked diffusion
- Mathematical connection to epistemic uncertainty theory
- Proposed experimental protocol (others can run on GPU)
- This is still publishable as a position/perspective paper
