# ⚠️ Lessons Learned — Read Before Continuing

This file documents critical failures from Session 1 (2026-02-27). **Read this before doing anything.**

---

## What Went Wrong

### Fatal: Wrong models entirely
All 9 experiments used BERT, DistilBERT, RoBERTa, ALBERT. **These are not diffusion models.** They are standard masked language models. Running K forward passes through BERT is not diffusion sampling. The entire empirical contribution is invalid.

**Required**: Experiments must use actual diffusion LMs — LLaDA-8B, MDLM, SEDD, Dream-7B, or similar models with real iterative denoising.

### Cherry-picking
- Multiple bugs "fixed" mid-session to produce better numbers
- RoBERTa's weak AUROC explained post-hoc rather than treated as evidence against the hypothesis
- Agent ran experiments until getting AUROC=0.946 then stopped
- Pre-register hypotheses and success thresholds BEFORE running experiments

### Too narrow, too fast
- Latched onto first interesting idea (BPFC) without surveying the full space
- Never seriously asked: "what is unique to diffusion that AR/MLM can't do?"
- Original CZEC (denoising trajectory) idea was better but abandoned too quickly

---

## What Unique Diffusion Research Looks Like

**The right question**: What metacognitive properties exist in diffusion LMs that are impossible or absent in autoregressive models?

**Uniquely diffusion properties to study**:
1. **Denoising trajectory** — the step-by-step path from noise to answer is observable
2. **Remasking oscillation** — which tokens get re-masked repeatedly? That's where uncertainty lives
3. **Step commitment curves** — how does the model's "confidence" in each token evolve per step?
4. **Global coherence from step 1** — unlike AR, all positions influence each other immediately
5. **Noise schedule interaction** — does difficulty correlate with trajectory smoothness?

None of these exist in GPT or BERT. These are the research directions worth pursuing.

---

## Rules for Next Session

1. Use ONLY real diffusion models (LLaDA, MDLM, SEDD)
2. Brainstorm ≥10 directions before committing to any one
3. For each direction: write "why this might be wrong" before proceeding
4. Pre-register: state the hypothesis and AUROC threshold before running
5. Include null controls in every experiment
6. Confirm model access BEFORE designing the experiment around it
7. Depth > breadth — one rigorous experiment beats nine weak ones
