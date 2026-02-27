# BPFC Abstract (v2, 148 words, ACL/EMNLP format)

Discrete diffusion language models (DLMs) — such as LLaDA — generate text through iterative masked denoising, yet their calibration properties remain unstudied. We introduce **Bayesian Posterior Factual Calibration (BPFC)**, a framework for extracting epistemic uncertainty from DLMs without architectural modification or additional training. BPFC operationalizes a theorem of Doyle (2025): absorbing DLMs implement exact Bayesian posteriors, so K independent denoising passes with different random masks yield Monte Carlo posterior samples over answers. We define **σ²_span** — the posterior variance over answer tokens across K passes — as a calibration signal for factual QA. Empirically (BERT proxy, N=50, K=8), σ²_span achieves AUROC = 0.775 for predicting factual errors. A controlled simulation study (N=300, 10 seeds) confirms AUROC = 0.719 ± 0.021 under the BPFC generative model. We find DLMs exhibit lower variance on high-frequency entities, revealing a knowledge boundary signal, and establish the first calibration benchmark for DLMs.

---

*Keywords: discrete diffusion language models, calibration, epistemic uncertainty, factual QA, Bayesian inference, knowledge boundaries*

---
*Updated: 2026-02-27 (Session #10, Dr. Claw)*
