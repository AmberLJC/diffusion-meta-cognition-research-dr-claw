#!/usr/bin/env python3
"""
BPFC Cross-Model Validation: RoBERTa-large (CORRECTED METHODOLOGY)
===================================================================
CRITICAL FIX: Previous roberta_crossval.py used word-dropout to create
K passes — this is WRONG. BPFC operationalization requires:
  1. Fixed cloze template with <mask> at answer position
  2. Temperature sampling from top-k distribution (K=8 passes)
  3. σ²_answer = variance of sampled confidence scores

This matches the BERT pilot methodology exactly, adapted for RoBERTa syntax.

Key methodological differences BERT ↔ RoBERTa:
  - [MASK] → <mask> (RoBERTa mask token)
  - RoBERTa is case-sensitive (no uncased variant)
  - Same temperature-sampling BPFC protocol

Author: Dr. Claw | 2026-02-27 Session #14
"""

import json
import math
import random
import statistics
import time
from pathlib import Path

random.seed(42)

# ── Cloze templates (adapted from BERT pilot) ─────────────────────────────────
# [MASK] → <mask>; answers kept in original case where possible

FACTUAL_QA = [
    # Easy (difficulty ~0.0–0.2)
    ("The capital of France is <mask>.", "Paris", 0.0),
    ("The capital of Germany is <mask>.", "Berlin", 0.0),
    ("The capital of Japan is <mask>.", "Tokyo", 0.0),
    ("The capital of Italy is <mask>.", "Rome", 0.0),
    ("The capital of Spain is <mask>.", "Madrid", 0.0),
    ("The capital of China is <mask>.", "Beijing", 0.0),
    ("Water is made of hydrogen and <mask>.", "oxygen", 0.0),
    ("The Earth orbits around the <mask>.", "Sun", 0.0),
    ("The largest ocean on Earth is the <mask> Ocean.", "Pacific", 0.0),
    ("Albert Einstein developed the theory of <mask>.", "relativity", 0.05),
    ("The chemical symbol for gold is <mask>.", "Au", 0.05),
    ("The Great Wall is located in <mask>.", "China", 0.05),
    ("The currency of Japan is the <mask>.", "Yen", 0.1),
    ("The Amazon River is located in <mask>.", "Brazil", 0.1),
    ("Leonardo da Vinci painted the Mona <mask>.", "Lisa", 0.05),
    ("Mount Everest is the tallest <mask> in the world.", "mountain", 0.05),
    ("Neil Armstrong was the first person to walk on the <mask>.", "Moon", 0.05),
    ("The chemical symbol for water is <mask>.", "H2O", 0.05),
    ("Shakespeare wrote Romeo and <mask>.", "Juliet", 0.05),
    ("The boiling point of water is <mask> degrees Celsius.", "100", 0.05),

    # Medium (difficulty ~0.3–0.6)
    ("The capital of Australia is <mask>.", "Canberra", 0.5),
    ("Charles Darwin proposed the theory of <mask>.", "evolution", 0.4),
    ("The Eiffel Tower is located in <mask>.", "Paris", 0.3),
    ("The chemical symbol for iron is <mask>.", "Fe", 0.4),
    ("Shakespeare wrote the play <mask>.", "Hamlet", 0.3),
    ("The powerhouse of the cell is the <mask>.", "mitochondria", 0.35),
    ("Isaac Newton discovered the law of <mask>.", "gravity", 0.3),
    ("The capital of Canada is <mask>.", "Ottawa", 0.5),
    ("The first element on the periodic table is <mask>.", "hydrogen", 0.3),
    ("The largest planet in our solar system is <mask>.", "Jupiter", 0.3),
    ("Penicillin was discovered by Alexander <mask>.", "Fleming", 0.4),
    ("The Sistine Chapel ceiling was painted by <mask>.", "Michelangelo", 0.4),
    ("The speed of light is approximately 300,000 km per <mask>.", "second", 0.4),
    ("The World Wide Web was invented by Tim Berners-<mask>.", "Lee", 0.45),
    ("The Berlin Wall fell in <mask>.", "1989", 0.4),

    # Hard (difficulty ~0.7–1.0)
    ("The capital of Kazakhstan is <mask>.", "Astana", 0.8),
    ("The chemical symbol for tungsten is <mask>.", "W", 0.8),
    ("The capital of Burkina Faso is <mask>.", "Ouagadougou", 1.0),
    ("Nikola Tesla was born in <mask>.", "Serbia", 0.75),
    ("The capital of Myanmar is <mask>.", "Naypyidaw", 0.9),
    ("Marie Curie was born in <mask>.", "Poland", 0.7),
    ("The capital of Iceland is <mask>.", "Reykjavik", 0.8),
    ("The Treaty of <mask> ended World War I.", "Versailles", 0.7),
    ("The Han dynasty ruled China from 206 BC to <mask> AD.", "220", 0.8),
    ("The Battle of <mask> was fought in 1815.", "Waterloo", 0.75),
    ("The smallest country in the world is <mask> City.", "Vatican", 0.7),
    ("The half-life of Carbon-14 is approximately <mask> years.", "5730", 0.85),
    ("The capital of Tajikistan is <mask>.", "Dushanbe", 0.85),
    ("The Chandrasekhar limit is approximately 1.4 solar <mask>.", "masses", 0.9),
    ("The Schwarzschild radius is proportional to the <mask> of a body.", "mass", 0.85),
]

N = len(FACTUAL_QA)
print(f"Dataset: N={N} questions")


def load_roberta():
    """Load RoBERTa-large fill-mask pipeline (CPU)."""
    print("Loading RoBERTa-large fill-mask pipeline (CPU)...")
    t0 = time.time()
    try:
        from transformers import pipeline as hf_pipeline
        mlm = hf_pipeline(
            "fill-mask",
            model="roberta-large",
            device=-1,
            top_k=50,
        )
        print(f"✅ RoBERTa-large loaded in {time.time()-t0:.1f}s")
        return mlm, "roberta-large"
    except Exception as e:
        print(f"RoBERTa-large failed ({e}), trying roberta-base...")
        try:
            from transformers import pipeline as hf_pipeline
            mlm = hf_pipeline(
                "fill-mask",
                model="roberta-base",
                device=-1,
                top_k=50,
            )
            print(f"✅ roberta-base loaded in {time.time()-t0:.1f}s")
            return mlm, "roberta-base"
        except Exception as e2:
            print(f"All RoBERTa variants failed: {e2}")
            return None, None


def sample_one_pass(pipe, template: str, temperature: float = 1.0):
    """
    Run one masked LM pass with temperature sampling.
    Returns (predicted_token, top1_confidence).
    
    Identical to BERT pilot methodology, just using <mask> token.
    """
    results = pipe(template)
    
    if isinstance(results[0], list):
        results = results[0]
    
    scores = [r["score"] for r in results]
    tokens = [r["token_str"].strip() for r in results]

    if temperature == 1.0:
        probs = scores
    else:
        log_probs = [math.log(max(s, 1e-10)) / temperature for s in scores]
        max_lp = max(log_probs)
        exp_lp = [math.exp(lp - max_lp) for lp in log_probs]
        total = sum(exp_lp)
        probs = [e / total for e in exp_lp]

    r_val = random.random()
    cumulative = 0.0
    chosen_idx = len(tokens) - 1
    for i, p in enumerate(probs):
        cumulative += p
        if r_val <= cumulative:
            chosen_idx = i
            break

    return tokens[chosen_idx], scores[chosen_idx]


def normalize(s: str) -> str:
    return s.lower().strip().rstrip(".,;:!?").replace("-", " ")


def bpfc_run(pipe, template: str, gold: str, K: int = 8, temperature: float = 1.0) -> dict:
    """Run K BPFC passes (temperature sampling) and compute σ²_answer."""
    predictions = []
    confidences = []

    for k in range(K):
        token, conf = sample_one_pass(pipe, template, temperature)
        predictions.append(token)
        confidences.append(conf)

    sigma2_answer = statistics.variance(confidences) if len(confidences) > 1 else 0.0
    majority_conf = statistics.mean(confidences)

    gold_n = normalize(gold)
    correct = any(
        gold_n in normalize(p) or normalize(p) in gold_n
        for p in predictions
    )

    return {
        "template": template,
        "gold": gold,
        "predictions": predictions,
        "confidences": confidences,
        "sigma2_answer": sigma2_answer,
        "majority_conf": majority_conf,
        "correct": correct,
    }


def compute_auroc(scores, labels):
    """AUROC: P(score[incorrect] > score[correct])."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    concordant = sum(1 for p in pos for n in neg if p > n)
    tied = sum(0.5 for p in pos for n in neg if p == n)
    return (concordant + tied) / (len(pos) * len(neg))


def bootstrap_auroc(scores, labels, B=2000):
    n = len(scores)
    vals = []
    for _ in range(B):
        idx = [random.randint(0, n - 1) for _ in range(n)]
        s = [scores[i] for i in idx]
        l = [labels[i] for i in idx]
        vals.append(compute_auroc(s, l))
    vals.sort()
    mean_ = statistics.mean(vals)
    lo = vals[int(0.025 * B)]
    hi = vals[int(0.975 * B)]
    return mean_, lo, hi


def main():
    print("=" * 60)
    print("BPFC Cross-Model Validation: RoBERTa (CORRECTED)")
    print("=" * 60)
    print("Methodology: same as BERT pilot (temperature sampling)")
    print(f"N={N}, K=8, temperature=1.0")
    print()

    pipe, model_name = load_roberta()
    if pipe is None:
        print("ERROR: Could not load RoBERTa. Exiting.")
        return

    results = []
    K = 8
    t_start = time.time()

    for i, (template, gold, diff) in enumerate(FACTUAL_QA):
        tier = "easy" if diff < 0.3 else ("medium" if diff < 0.65 else "hard")
        r = bpfc_run(pipe, template, gold, K=K, temperature=1.0)
        r["difficulty"] = diff
        r["tier"] = tier
        r["idx"] = i
        results.append(r)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            acc = sum(x["correct"] for x in results) / len(results)
            avg_s2 = statistics.mean(x["sigma2_answer"] for x in results)
            print(f"  [{i+1}/{N}] {elapsed:.0f}s | acc={acc:.2f} | avg_σ²={avg_s2:.5f}")

    elapsed = time.time() - t_start
    print(f"\n✅ Completed in {elapsed:.1f}s")

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    sigma2s = [r["sigma2_answer"] for r in results]
    maj_confs = [r["majority_conf"] for r in results]
    corrects = [r["correct"] for r in results]
    errors = [1 - int(c) for c in corrects]

    acc = sum(corrects) / N
    n_correct = sum(corrects)
    n_wrong = N - n_correct
    print(f"Accuracy: {acc:.3f} ({n_correct}/{N})")

    # AUROC: σ²_answer predicts error
    auroc_s2, lo_s2, hi_s2 = bootstrap_auroc(sigma2s, errors)
    print(f"\nσ²_answer AUROC: {auroc_s2:.3f} [{lo_s2:.3f}, {hi_s2:.3f}]")

    # AUROC: 1 - majority_conf predicts error
    inv_conf = [1 - c for c in maj_confs]
    auroc_mc, lo_mc, hi_mc = bootstrap_auroc(inv_conf, errors)
    print(f"majority_conf  AUROC: {auroc_mc:.3f} [{lo_mc:.3f}, {hi_mc:.3f}]")

    # Cohen's d (σ² separation)
    s2_correct = [r["sigma2_answer"] for r in results if r["correct"]]
    s2_wrong = [r["sigma2_answer"] for r in results if not r["correct"]]
    m_c = statistics.mean(s2_correct) if s2_correct else 0
    m_w = statistics.mean(s2_wrong) if s2_wrong else 0
    sd_c = statistics.stdev(s2_correct) if len(s2_correct) > 1 else 1e-9
    sd_w = statistics.stdev(s2_wrong) if len(s2_wrong) > 1 else 1e-9
    pooled = math.sqrt((sd_c ** 2 + sd_w ** 2) / 2)
    cohens_d = (m_w - m_c) / pooled if pooled > 1e-10 else 0.0
    print(f"\nMean σ² correct={m_c:.5f}, wrong={m_w:.5f}, Δ={m_w-m_c:.5f}")
    print(f"Cohen's d: {cohens_d:.3f}")

    # By tier
    print("\nTier breakdown:")
    for tier_name in ["easy", "medium", "hard"]:
        tier_r = [r for r in results if r["tier"] == tier_name]
        if not tier_r:
            continue
        t_acc = sum(r["correct"] for r in tier_r) / len(tier_r)
        t_s2 = statistics.mean(r["sigma2_answer"] for r in tier_r)
        t_mc = statistics.mean(r["majority_conf"] for r in tier_r)
        print(f"  {tier_name:8s}: acc={t_acc:.2f}, σ²={t_s2:.5f}, maj_conf={t_mc:.3f} (n={len(tier_r)})")

    # Pearson r (σ² vs difficulty)
    diffs = [r["difficulty"] for r in results]
    n = len(sigma2s)
    ms, md = statistics.mean(sigma2s), statistics.mean(diffs)
    cov = sum((sigma2s[i] - ms) * (diffs[i] - md) for i in range(n)) / n
    sds = math.sqrt(sum((x - ms) ** 2 for x in sigma2s) / n)
    sdd = math.sqrt(sum((x - md) ** 2 for x in diffs) / n)
    pearson_r = cov / (sds * sdd) if (sds * sdd) > 1e-10 else 0.0
    print(f"\nPearson r(σ², difficulty): {pearson_r:.3f}")

    # Cross-model comparison
    print("\n" + "=" * 60)
    print("CROSS-MODEL COMPARISON (BPFC σ²_answer AUROC)")
    print("=" * 60)
    print(f"{'Model':<22} {'AUROC':>12} {'95% CI':>22} {'Cohen_d':>10} {'N':>6}")
    print("-" * 75)
    print(f"{'BERT-base-uncased':<22} {'0.791':>12} {'[0.639, 0.927]':>22} {'1.626':>10} {'170':>6}")
    print(f"{model_name:<22} {auroc_s2:>12.3f} [{lo_s2:.3f}, {hi_s2:.3f}]  {cohens_d:>10.3f} {N:>6}")

    if auroc_s2 > 0.65:
        verdict = "✅ REPLICATED — BPFC signal generalizes across MLM architectures"
    elif auroc_s2 > 0.55:
        verdict = "⚠️  PARTIAL — weaker BPFC signal in RoBERTa (vocabulary / tokenization differences)"
    elif auroc_s2 > 0.45:
        verdict = "❓ MARGINAL — signal near chance; methodology difference may explain gap"
    else:
        verdict = "❌ NOT REPLICATED — σ²_answer below chance in RoBERTa"
    print(f"\nVerdict: {verdict}")

    # Save
    output = {
        "model": model_name,
        "methodology": "temperature_sampling_K8",
        "n": N,
        "k": 8,
        "accuracy": acc,
        "bpfc": {"auroc_mean": auroc_s2, "auroc_ci_lo": lo_s2, "auroc_ci_hi": hi_s2},
        "majority_conf": {"auroc_mean": auroc_mc, "auroc_ci_lo": lo_mc, "auroc_ci_hi": hi_mc},
        "cohens_d": cohens_d,
        "mean_sigma2_correct": m_c,
        "mean_sigma2_wrong": m_w,
        "pearson_r_difficulty": pearson_r,
        "tier_breakdown": {
            t: {
                "acc": sum(r["correct"] for r in results if r["tier"] == t) / max(1, len([r for r in results if r["tier"] == t])),
                "mean_sigma2": statistics.mean([r["sigma2_answer"] for r in results if r["tier"] == t] or [0]),
                "n": len([r for r in results if r["tier"] == t]),
            }
            for t in ["easy", "medium", "hard"]
        },
        "comparison": {
            "bert_auroc": 0.791,
            "roberta_auroc_corrected": auroc_s2,
            "replicated": auroc_s2 > 0.55,
        },
        "verdict": verdict,
    }

    out_path = Path("/home/azureuser/research/diffusion-meta-cognition-research-dr-claw/results/roberta_corrected_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")
    return output


if __name__ == "__main__":
    main()
