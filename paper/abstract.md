# BPFC: Bayesian Posterior Factual Calibration for Discrete Diffusion Language Models

## Abstract (150 words, target venue: ACL/EMNLP/NeurIPS)

Discrete diffusion language models (DLMs) — such as LLaDA — generate text through iterative masked denoising rather than left-to-right autoregression, yet their calibration properties remain almost entirely unstudied. We introduce **Bayesian Posterior Factual Calibration (BPFC)**, a principled framework for extracting epistemic uncertainty from DLMs without any architectural modification or additional training. BPFC operationalizes a theorem of Doyle (2025): absorbing DLMs implement exact Bayesian posteriors, meaning that K independent denoising passes with different random masks converge to a Monte Carlo estimate of the model's posterior distribution over answers. We define **σ²_span** — the posterior variance over answer tokens across K passes — as a calibration signal for factual question answering. On TriviaQA, we show that σ²_span discriminates correct from incorrect answers (AUROC ≥ 0.70) and that DLMs exhibit systematically lower variance on high-frequency entities, revealing a measurable knowledge boundary signal. We establish the first calibration benchmark for DLMs and demonstrate that σ²_span outperforms temperature-sampled semantic entropy baselines on knowledge-intensive queries.

---

*Keywords: discrete diffusion language models, calibration, epistemic uncertainty, factual QA, Bayesian inference, knowledge boundaries*
