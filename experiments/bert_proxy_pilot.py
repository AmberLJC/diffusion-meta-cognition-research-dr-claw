#!/usr/bin/env python3
"""
BPFC Proxy Experiment: BERT-based Masked Diffusion Uncertainty
==============================================================
CPU-feasible proxy for the full LLaDA BPFC experiment.

RATIONALE:
BERT (Devlin et al., 2019) is architecturally equivalent to a single-step
masked diffusion model: it assigns P(x_i | context) for masked positions.
We simulate K independent "denoising passes" by:
  1. Masking the answer position in a fill-in-the-blank QA format
  2. Sampling K times from BERT's softmax over the answer slot
  3. Computing σ²_answer = 1 - pairwise_agreement(K samples)
  4. Comparing to ground truth accuracy

This is a theoretically legitimate proxy because:
  - BERT's [MASK] token is the absorbing state of a 1-step diffusion process
  - P_BERT(x_i | context) approximates the Bayesian posterior over tokens
  - Multiple samples from this posterior estimate the posterior variance
  - The same BPFC metrics apply (AUROC, ECE, σ²_answer)

The proxy demonstrates BPFC works in principle; the full experiment uses
LLaDA-8B for sequence-level generation with iterative denoising.

DATASET: We use simple factual fill-in-the-blank templates:
  "The capital of [country] is [MASK]."
  "The [element_number] element is [MASK]."
  "[person] was born in [MASK]."

AUTHOR: Dr. Claw | 2026-02-27
"""

import json
import math
import random
import string
import statistics
import urllib.request
import urllib.parse
import time
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    "N_QUESTIONS": 30,
    "K_PASSES": 8,
    "TOP_K_TOKENS": 5,        # Sample from top-K tokens (simulates realistic sampling)
    "TEMPERATURE": 1.0,       # Sampling temperature
    "BERT_MODEL": "bert-base-uncased",
    "HF_ROUTER_URL": "https://router.huggingface.co/hf-inference/models",
    "OUTPUT_DIR": Path(__file__).parent.parent / "data",
    "RESULTS_FILE": "bert_proxy_results.jsonl",
    "ANALYSIS_FILE": "bert_proxy_analysis.json",
}

CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)

# ── Factual QA Dataset ─────────────────────────────────────────────────────── 

# Each entry: (template, correct_answer, difficulty)
# difficulty: 0=easy (frequent), 0.5=medium, 1.0=hard (rare)
FACTUAL_QA = [
    # Easy (very common facts)
    ("The capital of France is [MASK].", "paris", 0.0),
    ("The capital of Germany is [MASK].", "berlin", 0.0),
    ("Water is made of hydrogen and [MASK].", "oxygen", 0.0),
    ("The Earth orbits around the [MASK].", "sun", 0.0),
    ("The largest ocean on Earth is the [MASK] Ocean.", "pacific", 0.0),
    ("William Shakespeare was an English [MASK].", "playwright", 0.1),
    ("The chemical symbol for gold is [MASK].", "au", 0.1),
    ("Albert Einstein developed the theory of [MASK].", "relativity", 0.1),
    ("The Great Wall is located in [MASK].", "china", 0.1),
    ("The currency of Japan is the [MASK].", "yen", 0.1),
    
    # Medium
    ("The speed of light is approximately 300,000 kilometers per [MASK].", "second", 0.4),
    ("The capital of Australia is [MASK].", "canberra", 0.5),
    ("Charles Darwin proposed the theory of [MASK].", "evolution", 0.4),
    ("The Eiffel Tower is located in [MASK].", "paris", 0.3),
    ("The Amazon River is located in [MASK].", "brazil", 0.4),
    ("Marie Curie won two Nobel Prizes in Physics and [MASK].", "chemistry", 0.5),
    ("The first programmable computer was the [MASK].", "eniac", 0.6),
    ("The periodic table was created by Dmitri [MASK].", "mendeleev", 0.5),
    ("The Battle of Hastings was fought in [MASK].", "1066", 0.5),
    ("DNA stands for deoxyribonucleic [MASK].", "acid", 0.4),
    
    # Hard (obscure facts)
    ("The capital of Burkina Faso is [MASK].", "ouagadougou", 0.9),
    ("The Treaty of Westphalia was signed in [MASK].", "1648", 0.8),
    ("The longest river in Europe is the [MASK].", "volga", 0.7),
    ("The element with atomic number 92 is [MASK].", "uranium", 0.6),
    ("The Peloponnesian War was fought between Athens and [MASK].", "sparta", 0.7),
    ("The author of 'The Master and Margarita' is [MASK].", "bulgakov", 0.9),
    ("The Hagia Sophia was originally built as a [MASK].", "church", 0.7),
    ("The chemical formula for ammonia is [MASK].", "nh3", 0.7),
    ("The smallest country in the world by area is [MASK].", "vatican", 0.6),
    ("The programming language Python was created by Guido van [MASK].", "rossum", 0.8),
]


# ── BERT API Call ─────────────────────────────────────────────────────────────

def query_bert_fill_mask(text: str, top_k: int = 20, hf_token: str = "") -> Optional[list]:
    """
    Query BERT fill-mask via HF Router API.
    Returns list of {token_str, score} or None on failure.
    """
    import json, urllib.request, urllib.error
    
    url = f"{CONFIG['HF_ROUTER_URL']}/{CONFIG['BERT_MODEL']}"
    payload = json.dumps({"inputs": text, "parameters": {"top_k": top_k}}).encode()
    
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            if isinstance(result, list) and len(result) > 0:
                return result
            return None
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        print(f"    HTTP {e.code}: {body}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def temperature_sample(candidates: list, temperature: float, k: int) -> str:
    """
    Sample one token from candidates using temperature sampling.
    candidates: list of {token_str, score}
    """
    if not candidates:
        return "[UNK]"
    
    # Apply temperature to log-scores (scores are already probabilities)
    scores = [max(c.get("score", 0), 1e-10) for c in candidates[:k]]
    
    if temperature == 0:
        # Greedy
        return candidates[0]["token_str"].strip()
    
    # Temperature sampling
    log_scores = [math.log(s) / temperature for s in scores]
    # Softmax
    max_log = max(log_scores)
    exp_scores = [math.exp(l - max_log) for l in log_scores]
    total = sum(exp_scores)
    probs = [e / total for e in exp_scores]
    
    # Cumulative sampling
    r = random.random()
    cumul = 0.0
    for i, p in enumerate(probs):
        cumul += p
        if r <= cumul:
            return candidates[i]["token_str"].strip()
    return candidates[-1]["token_str"].strip()


# ── Metrics ───────────────────────────────────────────────────────────────────

def normalize(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.lower().strip()
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    answer = " ".join(answer.split())
    return answer


def pairwise_agreement(answers: list) -> float:
    """Fraction of pairs that agree after normalization."""
    normed = [normalize(a) for a in answers]
    k = len(normed)
    if k < 2:
        return 1.0
    count = 0
    total = 0
    for i in range(k):
        for j in range(i+1, k):
            total += 1
            if normed[i] == normed[j]:
                count += 1
    return count / total if total > 0 else 1.0


def is_correct(prediction: str, gold: str) -> bool:
    """Check if prediction matches gold answer."""
    return normalize(prediction) == normalize(gold)


def compute_auroc(scores: list, labels: list) -> float:
    """
    Compute AUROC where higher score = more uncertain = more likely incorrect.
    labels[i] = 1 if incorrect, 0 if correct.
    """
    n1 = sum(labels)     # positives (incorrect)
    n0 = len(labels) - n1  # negatives (correct)
    if n0 == 0 or n1 == 0:
        return 0.5
    
    # Sort by score descending
    pairs = sorted(zip(scores, labels), reverse=True)
    
    # Wilcoxon rank-sum
    rank_sum = 0
    rank = 1
    for score, label in pairs:
        if label == 1:
            rank_sum += rank
        rank += 1
    
    auroc = (rank_sum - n1 * (n1 + 1) / 2) / (n0 * n1)
    return auroc


def compute_ece(scores: list, labels: list, n_bins: int = 5) -> float:
    """
    Expected Calibration Error.
    Confidence = 1 - sigma2_answer (higher sigma2 → less confident)
    """
    n = len(scores)
    if n == 0:
        return 0.0
    
    # Bin by confidence (1 - score)
    confidences = [1.0 - s for s in scores]
    correct = [1 - l for l in labels]  # 1 if correct
    
    # Equal-frequency bins
    sorted_by_conf = sorted(zip(confidences, correct))
    bin_size = max(1, n // n_bins)
    
    ece = 0.0
    for b in range(n_bins):
        start = b * bin_size
        end = start + bin_size if b < n_bins - 1 else n
        bin_items = sorted_by_conf[start:end]
        if not bin_items:
            continue
        avg_conf = sum(c for c, _ in bin_items) / len(bin_items)
        avg_acc = sum(c for _, c in bin_items) / len(bin_items)
        ece += (len(bin_items) / n) * abs(avg_conf - avg_acc)
    
    return ece


def pearson_r(xs: list, ys: list) -> float:
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den = (sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys)) ** 0.5
    return num/den if den > 0 else 0.0


# ── Main Experiment ───────────────────────────────────────────────────────────

def run_question(q_idx: int, template: str, gold: str, difficulty: float, hf_token: str = "") -> dict:
    """Run K passes for one question and compute BPFC metrics."""
    print(f"\n  Q{q_idx+1}: {template[:60]}...")
    print(f"  Gold: '{gold}', Difficulty: {difficulty:.1f}")
    
    # Query BERT once (all K samples come from the same distribution)
    candidates = query_bert_fill_mask(template, top_k=20, hf_token=hf_token)
    
    if not candidates:
        print(f"  ❌ API failed for Q{q_idx+1}")
        return {
            "idx": q_idx, "template": template, "gold": gold, "difficulty": difficulty,
            "answers": [], "sigma2": 0.5, "correct": False, "api_error": True
        }
    
    print(f"  Top-3 BERT tokens: {[(c['token_str'].strip(), f\"{c['score']:.3f}\") for c in candidates[:3]]}")
    
    # K independent samples (same distribution = K independent denoising passes)
    answers = []
    for k in range(CONFIG["K_PASSES"]):
        sampled = temperature_sample(candidates, CONFIG["TEMPERATURE"], CONFIG["TOP_K_TOKENS"])
        answers.append(sampled)
    
    # BPFC metrics
    agreement = pairwise_agreement(answers)
    sigma2 = 1.0 - agreement
    
    # Greedy prediction (modal answer for accuracy)
    from collections import Counter
    modal = Counter(normalize(a) for a in answers).most_common(1)[0][0]
    correct = is_correct(modal, gold)
    
    print(f"  Answers: {answers}")
    print(f"  σ²_answer = {sigma2:.3f}, Correct = {correct}")
    
    return {
        "idx": q_idx, "template": template, "gold": gold, "difficulty": difficulty,
        "answers": answers, "agreement": agreement, "sigma2": sigma2,
        "modal_answer": modal, "correct": correct, "api_error": False,
        "top_candidates": [(c["token_str"].strip(), round(c["score"], 4)) for c in candidates[:5]]
    }


def main():
    import os
    hf_token = os.environ.get("HF_TOKEN", "")
    
    print("=" * 60)
    print("BPFC Proxy Experiment: BERT Masked Diffusion Uncertainty")
    print("=" * 60)
    
    random.seed(CONFIG.get("SEED", 42))
    
    # Sample N questions
    qa = FACTUAL_QA[:CONFIG["N_QUESTIONS"]]
    
    results = []
    errors = 0
    
    for i, (template, gold, difficulty) in enumerate(qa):
        result = run_question(i, template, gold, difficulty, hf_token)
        results.append(result)
        
        if result.get("api_error"):
            errors += 1
        
        # Save incrementally
        with open(CONFIG["OUTPUT_DIR"] / CONFIG["RESULTS_FILE"], "a") as f:
            f.write(json.dumps(result) + "\n")
        
        time.sleep(0.5)  # Rate limiting
    
    # ── Analysis ──────────────────────────────────────────────────────────────
    valid = [r for r in results if not r.get("api_error")]
    n = len(valid)
    
    if n < 5:
        print(f"\n❌ Too few valid results ({n}), analysis aborted")
        return
    
    scores = [r["sigma2"] for r in valid]
    labels = [0 if r["correct"] else 1 for r in valid]  # 1 = incorrect
    diffs = [r["difficulty"] for r in valid]
    
    auroc = compute_auroc(scores, labels)
    ece = compute_ece(scores, labels)
    accuracy = sum(r["correct"] for r in valid) / n
    mean_sigma2 = sum(scores) / n
    corr_diff = pearson_r(scores, diffs)  # σ² vs difficulty
    
    analysis = {
        "n_total": len(results),
        "n_valid": n,
        "n_errors": errors,
        "k_passes": CONFIG["K_PASSES"],
        "accuracy": round(accuracy, 4),
        "mean_sigma2": round(mean_sigma2, 4),
        "auroc": round(auroc, 4),
        "ece": round(ece, 4),
        "pearson_sigma2_vs_difficulty": round(corr_diff, 4),
        "by_difficulty_tercile": {}
    }
    
    # Stratify by difficulty
    easy = [r for r in valid if r["difficulty"] <= 0.3]
    medium = [r for r in valid if 0.3 < r["difficulty"] <= 0.6]
    hard = [r for r in valid if r["difficulty"] > 0.6]
    
    for tier, group in [("easy", easy), ("medium", medium), ("hard", hard)]:
        if group:
            analysis["by_difficulty_tercile"][tier] = {
                "n": len(group),
                "accuracy": round(sum(r["correct"] for r in group) / len(group), 3),
                "mean_sigma2": round(sum(r["sigma2"] for r in group) / len(group), 3),
            }
    
    with open(CONFIG["OUTPUT_DIR"] / CONFIG["ANALYSIS_FILE"], "w") as f:
        json.dump(analysis, f, indent=2)
    
    # ── Print Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  N = {n} questions, K = {CONFIG['K_PASSES']} passes each")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Mean σ²_answer: {mean_sigma2:.3f}")
    print(f"  AUROC (σ² vs error): {auroc:.3f}  (>0.5 = calibrated)")
    print(f"  ECE: {ece:.3f}  (<0.1 = well-calibrated)")
    print(f"  Pearson(σ², difficulty): {corr_diff:.3f}  (>0 = expected)")
    print()
    print("  Difficulty stratification:")
    for tier, stats in analysis["by_difficulty_tercile"].items():
        print(f"    {tier}: acc={stats['accuracy']:.1%}, σ²={stats['mean_sigma2']:.3f} (n={stats['n']})")
    
    print(f"\n✅ Results saved to {CONFIG['OUTPUT_DIR']}")


if __name__ == "__main__":
    main()
