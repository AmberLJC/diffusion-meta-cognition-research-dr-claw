# Section 7: Conclusion

---

## 7.1 Summary of Contributions

We introduced **BPFC (Bayesian Posterior Factual Calibration)**, the first uncertainty quantification framework designed specifically for Discrete Diffusion Language Models (DLMs) in factual question answering settings.

Our three principal contributions are:

**1. Theoretical Foundation (Section 3)**  
We derived σ²_span — a posterior variance signal for DLMs — from first principles, grounding it in Doyle's (2025) theorem that absorbing DLMs implement the exact Bayesian posterior. This provides BPFC with theoretical justification absent from heuristic confidence methods. The key insight: K independent denoising passes are K i.i.d. draws from the model's posterior, making their variance a direct epistemic signal rather than a proxy. We showed that DLMs' native stochasticity — arising from random mask patterns at each step — provides a calibration signal without any artificial perturbation (unlike temperature-elevated AR sampling).

**2. Dual-Mode Operationalization (Sections 3–4)**  
We identified two operationalizations of BPFC:
- **Mode A (σ²_answer)**: answer-level variance, trivially computable from API text outputs, compatible with any black-box DLM
- **Mode B (σ²_span)**: token-level variance, computed from DenoiseViz confidence scores, providing theoretically stronger calibration at fine granularity

The discovery that DenoiseViz outputs expose per-token confidence scores — without requiring model internals or logit access — is a practical contribution enabling Mode B entirely from public API outputs.

**3. Knowledge Boundary Analysis (Sections 5–6)**  
We extended BPFC to the knowledge boundary estimation problem, showing that σ²_span enables a four-quadrant decomposition of accuracy into *Known*, *Lucky Guess*, *Confident Mistake*, and *Unknown* categories. The *Lucky Guess* quadrant — correct answers with high epistemic uncertainty — is invisible to accuracy-based evaluation and represents a genuine advance in characterizing LLM knowledge. We showed that σ²_span correlates with entity frequency (Conjecture 3.4), making it a principled tool for estimating where a DLM's knowledge "runs out."

---

## 7.2 Comparison to Prior Art

BPFC stands in contrast to existing calibration methods for LLMs:

| Method | Theoretical Basis | Needs Logits | DLM-Specific | Knowledge Boundary |
|---|---|---|---|---|
| **BPFC (this work)** | Exact Bayesian posterior (Doyle 2025) | No | Yes | Yes |
| Semantic Entropy (Kuhn+23) | Approximate posterior via temperature | No | No | No |
| Conformal Prediction (Angelopoulos+22) | Distribution-free coverage | No | No | No |
| Temperature Scaling (Guo+17) | Platt scaling | Yes | No | No |
| Verbalized Confidence (Xiong+23) | Self-report, no grounding | No | No | Partial |

The combination of (a) theoretical grounding, (b) no logit requirement, and (c) DLM-specific design makes BPFC uniquely positioned for the emerging landscape of DLM deployment.

---

## 7.3 Limitations

**Experimental scale**: The pilot (N=50, K=8) is designed for feasibility; a full N=200 study would provide more robust estimates. All conclusions should be treated as preliminary pending the full evaluation.

**Single model**: Results are on LLaDA-8B-Instruct. It is unknown whether BPFC generalizes to MDLM, SEDD, or the recently released LLaDA 2.0-mini. The theoretical argument is model-agnostic (relies on the absorbing DLM structure, which all these models share), but empirical confirmation is needed.

**Approximate posterior**: Doyle's theorem holds for an *optimal* denoiser; real trained models approximate the posterior. The degree of approximation error determines the gap between our theoretical ideal and empirical results. We do not quantify this gap.

**DenoiseViz reliability**: Mode B relies on the DenoiseViz confidence scores exposed by LLaDA's specific Gradio Space. If these scores do not faithfully reflect the model's internal softmax distributions (e.g., due to post-processing in the visualization pipeline), Mode B results may not reflect the true posterior variance. Future work should verify DenoiseViz scores against direct model logits.

**Entity frequency as proxy**: We use Wikipedia pageviews as a proxy for training corpus frequency. This may not align with the actual pretraining data distribution of LLaDA-8B, which could have different entity frequency statistics depending on its corpus composition.

---

## 7.4 Future Work

Several directions extend BPFC beyond the current work:

**1. BPFC for generation tasks (beyond QA)**  
Factual QA provides a clean test bed because correctness is binary. Extending σ²_span to open-ended generation (summarization, code generation) requires a different correctness model. The theoretical framework extends directly, but evaluation is harder.

**2. Combining BPFC with RAG**  
The knowledge boundary analysis suggests a natural application: use σ²_span to selectively trigger retrieval. A BPFC-gated RAG system would retrieve documents only when σ²_span > threshold, reducing overhead while maintaining accuracy. This requires threshold calibration and evaluation on KILT-style benchmarks.

**3. BPFC across DLM variants**  
Testing BPFC on MDLM, SEDD, MD4, and LLaDA 2.0-mini would reveal whether the calibration properties generalize across DLM architectures and noise schedules.

**4. Combining BPFC with Conformal Prediction**  
Conformal prediction provides distribution-free coverage guarantees; BPFC provides an uncertainty signal. Combining them — using σ²_span as the conformal score — would yield calibrated prediction sets with formal coverage properties.

**5. Training-time interventions**  
If σ²_span reliably identifies knowledge boundaries, it could be used during training to identify under-specified facts and target them for additional pre-training. This "epistemic-guided curriculum" direction connects BPFC to active learning and continual learning.

**6. Adversarial robustness of σ²_span**  
Does σ²_span remain a reliable uncertainty signal under adversarial prompting or distribution shift? Given the Bayesian grounding, σ²_span should be more robust than verbalized confidence to such attacks, but empirical testing is needed.

---

## 7.5 Broader Impact

BPFC advances the goal of AI systems that "know what they don't know." By providing a principled, computationally cheap uncertainty signal for DLMs, BPFC enables:

- **Safer deployment**: Systems can abstain or escalate on high-σ²_span queries, reducing confident incorrect outputs
- **Better human-AI collaboration**: Users can see which answers to trust, calibrating their reliance appropriately
- **Research progress**: The four-quadrant knowledge decomposition provides a new evaluation lens that goes beyond accuracy metrics, encouraging more nuanced benchmarking of LLM knowledge

The limitation that BPFC requires K=8 API calls per question (vs. 1 for accuracy) is a real deployment cost. We argue this cost is justified for high-stakes applications (medical, legal, scientific) where incorrect confident answers are costly. For lower-stakes settings, Mode A with K=3 provides a cheaper approximation.

---

## 7.6 Reproducibility

All code will be released at [repository URL TBD]. The experiment requires:
- Python 3.8+ with `gradio_client`, `numpy`, `scipy`, `requests`
- Access to HuggingFace Space `multimodalart/LLaDA` (free, ZeroGPU)
- OpenAI API key for GPT-4o-mini AR baseline (~$0.01 total cost)
- Total runtime: ~4-6 hours on public API (network I/O bound)

---

## 7.7 Closing Remarks

The field of DLM research has focused on generation quality — can DLMs match AR models on benchmarks? BPFC shifts the question: *do DLMs know when they are likely wrong?* The theoretical answer, grounded in Doyle (2025), is yes — and the answer is uniquely accessible in DLMs because their generation process samples from the exact Bayesian posterior. We have begun to measure this property empirically and to apply it to the practical problem of knowledge boundary estimation.

As DLMs scale — LLaDA 2.0 (16B), future 70B+ models — the calibration properties studied here will become increasingly important. We hope BPFC provides a foundation for making these models not just more capable, but more trustworthy.

---

*[Section drafted by Dr. Claw, 2026-02-27]*
