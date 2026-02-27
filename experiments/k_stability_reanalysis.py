#!/usr/bin/env python3
"""
K-Stability Re-Analysis: BPFC AUROC vs K using CORRECT σ²_answer metric
==========================================================================
BUGFIX: The previous k_stability_analysis.py computed σ²_answer as the
variance of binary [correct/incorrect] labels (which uses the gold answer
and is NOT the BPFC signal). This re-analysis uses the CORRECT metric:
    σ²_answer = 1 - mean_pairwise_agreement (fraction of K*(K-1)/2 pairs
                that give DIFFERENT answer tokens)
This does NOT require the gold label — it is the proper epistemic signal.

We subsample K=1..16 from the existing K=8 bert_cpu_results.jsonl (N=50)
using random subsets of passes to estimate convergence.

AUTHOR: Dr. Claw | 2026-02-27 (Bugfix session)
"""

import json
import math
import random
import statistics
from pathlib import Path

INPUT_FILE = Path(__file__).parent.parent / "data" / "bert_cpu_results.jsonl"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "k_stability_corrected.json"

random.seed(42)
N_BOOTSTRAP = 50  # Bootstrap resamples for K < K_max


def normalize_answer(ans: str) -> str:
    ans = ans.lower().strip()
    for article in ["the ", "a ", "an "]:
        if ans.startswith(article):
            ans = ans[len(article):]
    return ans.strip()


def compute_sigma2_answer_correct(answers: list[str]) -> float:
    """CORRECT BPFC metric: pairwise token disagreement (gold-label-free)."""
    K = len(answers)
    if K <= 1:
        return 0.0
    norms = [normalize_answer(a) for a in answers]
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    disagreements = sum(1 for i, j in pairs if norms[i] != norms[j])
    return disagreements / len(pairs)


def compute_sigma2_answer_buggy(answers: list[str], gold: str) -> float:
    """BUGGY metric from k_stability_analysis.py: variance of [correct/incorrect]."""
    K = len(answers)
    if K <= 1:
        return 0.0
    norms = [normalize_answer(a) for a in answers]
    gold_norm = normalize_answer(gold)
    pass_correct = [int(a == gold_norm) for a in norms]
    mean_pc = sum(pass_correct) / K
    return sum((x - mean_pc) ** 2 for x in pass_correct) / K


def auroc(scores: list, labels: list) -> float:
    """AUROC. labels=1 → positive class (error/incorrect)."""
    pos = [(s, True) for s, l in zip(scores, labels) if l]
    neg = [(s, False) for s, l in zip(scores, labels) if not l]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Count: for each pos/neg pair, score[pos] > score[neg]
    count = sum(1 for s_p, _ in pos for s_n, _ in neg if s_p > s_n)
    count += sum(0.5 for s_p, _ in pos for s_n, _ in neg if s_p == s_n)
    return count / (n_pos * n_neg)


# Load N=50 results
print(f"Loading {INPUT_FILE}...")
results = []
with open(INPUT_FILE) as f:
    for line in f:
        r = json.loads(line)
        results.append(r)
print(f"Loaded {len(results)} questions (K=8 each)")

# Available K_max from data
K_MAX_DATA = len(results[0]["dlm_answers"])  # should be 8
K_TEST = [1, 2, 3, 4, 5, 6, 7, 8]

print(f"\nK_MAX in data: {K_MAX_DATA}")
print(f"Testing K values: {K_TEST}")
print(f"Bootstrap resamples per K: {N_BOOTSTRAP}")

k_stability_data = []
for k in K_TEST:
    if k == K_MAX_DATA:
        # Use full data, no resampling needed
        sigma2_vals = [
            compute_sigma2_answer_correct(r["dlm_answers"][:k]) for r in results
        ]
        sigma2_buggy = [
            compute_sigma2_answer_buggy(r["dlm_answers"][:k], r["gold"]) for r in results
        ]
        labels = [1 - int(r["is_correct"]) for r in results]
        auroc_correct = auroc(sigma2_vals, labels)
        auroc_buggy_ = auroc(sigma2_buggy, labels)
        accuracy = sum(r["is_correct"] for r in results) / len(results)
        auroc_std = 0.0
    else:
        # Bootstrap: take N_BOOTSTRAP different random subsets of K passes
        auroc_samples = []
        for _ in range(N_BOOTSTRAP):
            sigma2_bs = []
            labels_bs = []
            for r in results:
                # Random subsample of K passes
                idxs = random.sample(range(K_MAX_DATA), k)
                sub_answers = [r["dlm_answers"][i] for i in idxs]
                sigma2_bs.append(compute_sigma2_answer_correct(sub_answers))
                labels_bs.append(1 - int(r["is_correct"]))
            auroc_samples.append(auroc(sigma2_bs, labels_bs))
        auroc_correct = statistics.mean(auroc_samples)
        auroc_std = statistics.stdev(auroc_samples)
        accuracy = sum(r["is_correct"] for r in results) / len(results)
        auroc_buggy_ = None  # not computed for subsampled

    entry = {
        "k": k,
        "n": len(results),
        "accuracy": accuracy,
        "auroc_sigma2_answer_CORRECT": round(auroc_correct, 4),
        "auroc_std": round(auroc_std, 4) if auroc_std else 0.0,
        "auroc_sigma2_answer_BUGGY": round(auroc_buggy_, 4) if auroc_buggy_ is not None else None,
    }
    k_stability_data.append(entry)
    print(f"  K={k:2d} | AUROC(σ²_answer_CORRECT)={auroc_correct:.4f} ± {auroc_std:.4f} | "
          f"acc={accuracy:.2f}")

# Summary
print("\n── K-Stability Table (CORRECTED METRIC) ───────────────")
print(f"{'K':>4}  {'AUROC':>10}  {'±':>8}")
print("-" * 30)
for r in k_stability_data:
    print(f"{r['k']:>4}  {r['auroc_sigma2_answer_CORRECT']:>10.4f}  "
          f"{r['auroc_std']:>7.4f}")

# Check for plateau at K≥4
high_k = [r['auroc_sigma2_answer_CORRECT'] for r in k_stability_data if r['k'] >= 4]
spread = max(high_k) - min(high_k) if high_k else 0.0
print(f"\nAUROC spread for K≥4: {spread:.4f}")
print(f"Mean AUROC for K≥4: {statistics.mean(high_k):.4f}")

if spread < 0.05:
    print("✅ CONVERGED: spread < 0.05 — K=4 is sufficient")
else:
    print(f"⚠️ Not fully converged: spread = {spread:.4f}")

# Bug explanation
print("\n── Bug Analysis ─────────────────────────────────────────")
buggy_entry = next(r for r in k_stability_data if r['k'] == K_MAX_DATA)
print(f"BUGGY σ²_answer (K={K_MAX_DATA}, gold-dependent): AUROC = {buggy_entry['auroc_sigma2_answer_BUGGY']:.4f}")
print(f"CORRECT σ²_answer (K={K_MAX_DATA}, gold-free): AUROC = {buggy_entry['auroc_sigma2_answer_CORRECT']:.4f}")
print()
print("The buggy metric measures variance of [correct/incorrect] labels.")
print("For easy questions (always correct), this = 0 (correct).")
print("For hard questions (always wrong), this = 0 (incorrect).")
print("→ Both correct AND incorrect questions can have σ²=0 → anti-calibrated.")
print()
print("The correct metric measures pairwise answer token disagreement.")
print("For easy questions (confident): low variance → correct → well calibrated.")
print("For hard questions (confused): high variance → incorrect → well calibrated.")

# Save
output = {
    "k_results": k_stability_data,
    "bug_explanation": (
        "Previous k_stability_analysis.py computed σ²_answer = variance of binary "
        "correct/incorrect labels, which requires the gold answer and is anti-calibrated "
        "for hard questions (always wrong → σ²=0 regardless of error). "
        "Correct BPFC metric = pairwise answer token disagreement (gold-free), "
        "which showed AUROC=0.775 in the original bert_cpu_pilot.py."
    ),
    "correct_metric_at_k8": buggy_entry['auroc_sigma2_answer_CORRECT'],
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n✓ Saved to {OUTPUT_FILE}")
