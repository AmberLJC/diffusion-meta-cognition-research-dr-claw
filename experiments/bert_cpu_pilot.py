#!/usr/bin/env python3
"""
BPFC CPU Pilot: BERT-based Masked Diffusion Uncertainty (LOCAL CPU)
====================================================================
Runs entirely on CPU using huggingface transformers + BERT-base.
No API key, no GPU, no quota limits.

AUTHOR: Dr. Claw | 2026-02-27 (Run #6)
"""

import json
import math
import random
import string
import statistics
import time
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    "N_QUESTIONS": 50,
    "K_PASSES": 8,
    "BERT_MODEL": "bert-base-uncased",
    "TEMPERATURE": 1.0,
    "SEED": 42,
    "OUTPUT_DIR": Path(__file__).parent.parent / "data",
    "RESULTS_FILE": "bert_cpu_results.jsonl",
    "ANALYSIS_FILE": "bert_cpu_analysis.json",
}
CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
random.seed(CONFIG["SEED"])

# ── Factual QA Dataset ─────────────────────────────────────────────────────── 

# Each entry: (fill_template, correct_answer, difficulty 0=easy→1=hard)
# Template must have exactly one [MASK] at the answer position
FACTUAL_QA = [
    # === EASY (difficulty 0.0–0.2, very common facts) ===
    ("The capital of France is [MASK].", "paris", 0.0),
    ("The capital of Germany is [MASK].", "berlin", 0.0),
    ("The capital of Japan is [MASK].", "tokyo", 0.0),
    ("The capital of Italy is [MASK].", "rome", 0.0),
    ("The capital of Spain is [MASK].", "madrid", 0.0),
    ("The capital of China is [MASK].", "beijing", 0.0),
    ("Water is made of hydrogen and [MASK].", "oxygen", 0.0),
    ("The Earth orbits around the [MASK].", "sun", 0.0),
    ("The largest ocean on Earth is the [MASK] Ocean.", "pacific", 0.0),
    ("Albert Einstein developed the theory of [MASK].", "relativity", 0.05),
    ("The chemical symbol for gold is [MASK].", "au", 0.05),
    ("William Shakespeare was an English [MASK].", "playwright", 0.1),
    ("The Great Wall is located in [MASK].", "china", 0.05),
    ("The currency of Japan is the [MASK].", "yen", 0.1),
    ("Neil Armstrong was the first person to walk on the [MASK].", "moon", 0.05),
    ("The Amazon River is located in [MASK].", "brazil", 0.1),
    ("The chemical symbol for water is [MASK].", "h2o", 0.05),
    ("Leonardo da Vinci painted the Mona [MASK].", "lisa", 0.05),
    ("The speed of sound travels through [MASK].", "air", 0.1),
    ("Mount Everest is the tallest [MASK] in the world.", "mountain", 0.05),
    
    # === MEDIUM (difficulty 0.3–0.6) ===
    ("The capital of Australia is [MASK].", "canberra", 0.5),
    ("Charles Darwin proposed the theory of [MASK].", "evolution", 0.4),
    ("The speed of light is approximately 300,000 kilometers per [MASK].", "second", 0.4),
    ("The Eiffel Tower is located in [MASK].", "paris", 0.3),
    ("The chemical symbol for iron is [MASK].", "fe", 0.4),
    ("The human body has [MASK] bones.", "206", 0.4),
    ("Shakespeare wrote the play [MASK].", "hamlet", 0.3),
    ("The powerhouse of the cell is the [MASK].", "mitochondria", 0.35),
    ("Isaac Newton discovered the law of [MASK].", "gravity", 0.3),
    ("The capital of Canada is [MASK].", "ottawa", 0.5),
    ("The [MASK] War lasted from 1939 to 1945.", "second", 0.3),
    ("The first element on the periodic table is [MASK].", "hydrogen", 0.3),
    ("DNA stands for deoxyribonucleic [MASK].", "acid", 0.35),
    ("The theory of relativity was developed by [MASK].", "einstein", 0.3),
    ("The largest planet in our solar system is [MASK].", "jupiter", 0.3),
    
    # === HARD (difficulty 0.7–1.0, obscure facts) ===
    ("The capital of Kazakhstan is [MASK].", "astana", 0.8),
    ("The chemical symbol for tungsten is [MASK].", "w", 0.8),
    ("The Treaty of [MASK] ended World War I.", "versailles", 0.7),
    ("The capital of Burkina Faso is [MASK].", "ouagadougou", 1.0),
    ("The [MASK] dynasty ruled China from 206 BC to 220 AD.", "han", 0.8),
    ("Nikola Tesla was born in [MASK].", "serbia", 0.75),
    ("The chemical formula for table salt is [MASK].", "nacl", 0.7),
    ("The capital of Myanmar is [MASK].", "naypyidaw", 0.9),
    ("The Battle of [MASK] was fought in 1815.", "waterloo", 0.75),
    ("The smallest country in the world is [MASK] City.", "vatican", 0.7),
    ("Marie Curie was born in [MASK].", "poland", 0.7),
    ("The [MASK] Ocean is the smallest ocean.", "arctic", 0.75),
    ("The capital of Iceland is [MASK].", "reykjavik", 0.8),
    ("The Pythagorean theorem relates the sides of a right [MASK].", "triangle", 0.4),
    ("The inventor of the telephone was Alexander Graham [MASK].", "bell", 0.3),
]

assert len(FACTUAL_QA) >= CONFIG["N_QUESTIONS"], f"Need {CONFIG['N_QUESTIONS']} questions, have {len(FACTUAL_QA)}"
FACTUAL_QA = FACTUAL_QA[:CONFIG["N_QUESTIONS"]]

# ── BERT Inference ────────────────────────────────────────────────────────────

def load_bert():
    """Load BERT fill-mask pipeline (downloads ~440MB on first run)."""
    print("Loading BERT model (CPU)...")
    t0 = time.time()
    from transformers import pipeline
    pipe = pipeline(
        "fill-mask",
        model=CONFIG["BERT_MODEL"],
        device=-1,  # CPU
        top_k=50,   # Get top-50 tokens for sampling
    )
    print(f"BERT loaded in {time.time()-t0:.1f}s")
    return pipe


def sample_one_pass(pipe, template: str, temperature: float = 1.0) -> tuple[str, float]:
    """
    Run one BERT fill-mask pass on the template.
    Returns (predicted_token, confidence).
    """
    results = pipe(template)  # returns list of {token_str, score, ...}
    
    # Apply temperature to logits (log(score) / temp → re-normalize)
    scores = [r["score"] for r in results]
    tokens = [r["token_str"].strip().lower() for r in results]
    
    if temperature == 1.0:
        probs = scores
    else:
        log_probs = [math.log(max(s, 1e-10)) / temperature for s in scores]
        max_lp = max(log_probs)
        exp_lp = [math.exp(lp - max_lp) for lp in log_probs]
        total = sum(exp_lp)
        probs = [e / total for e in exp_lp]
    
    # Sample token
    r = random.random()
    cumulative = 0.0
    chosen_idx = len(tokens) - 1
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            chosen_idx = i
            break
    
    return tokens[chosen_idx], scores[chosen_idx]


def normalize_answer(s: str) -> str:
    """Normalize answer string (lowercase, strip punct/articles)."""
    s = s.lower().strip()
    s = ''.join(c for c in s if c not in string.punctuation)
    s = ' '.join(s.split())
    for art in ['the', 'a', 'an']:
        if s.startswith(art + ' '):
            s = s[len(art)+1:]
    return s.strip()


def check_correct(predicted: str, gold: str) -> bool:
    """Check if predicted matches gold (with normalization)."""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    # Exact match
    if pred_norm == gold_norm:
        return True
    # Partial match (pred is contained in gold or vice versa)
    if pred_norm in gold_norm or gold_norm in pred_norm:
        return True
    return False


def compute_sigma2_answer(answers: list[str]) -> float:
    """
    Compute answer-level variance: 1 - mean_pairwise_agreement.
    Returns value in [0, 1].
    """
    K = len(answers)
    if K <= 1:
        return 0.0
    pairs = [(i, j) for i in range(K) for j in range(i+1, K)]
    agreements = sum(
        1 for i, j in pairs
        if normalize_answer(answers[i]) == normalize_answer(answers[j])
    )
    return 1.0 - agreements / len(pairs)


def compute_sigma2_token(confidences: list[float]) -> float:
    """
    Compute token-level confidence variance across K passes.
    This is the Mode B σ²_span signal.
    Returns sample variance of the K confidence scores.
    """
    if len(confidences) <= 1:
        return 0.0
    mean_c = statistics.mean(confidences)
    return statistics.variance(confidences)  # uses K-1 denominator


# ── Main Experiment ───────────────────────────────────────────────────────────

def run_experiment():
    pipe = load_bert()
    
    results = []
    
    for idx, (template, gold, difficulty) in enumerate(FACTUAL_QA):
        q_id = f"q{idx:03d}"
        print(f"\n[{idx+1}/{CONFIG['N_QUESTIONS']}] {template[:60]}...", end="", flush=True)
        
        answers = []
        confidences = []
        
        for k in range(CONFIG["K_PASSES"]):
            token, conf = sample_one_pass(pipe, template, CONFIG["TEMPERATURE"])
            answers.append(token)
            confidences.append(conf)
        
        # Mode A: answer-level variance
        sigma2_answer = compute_sigma2_answer(answers)
        # Mode B: token-level confidence variance (σ²_span proxy)
        sigma2_token = compute_sigma2_token(confidences)
        
        # Correctness via majority vote (most common answer)
        from collections import Counter
        vote_counter = Counter(normalize_answer(a) for a in answers)
        majority_answer = vote_counter.most_common(1)[0][0]
        is_correct = check_correct(majority_answer, gold)
        
        # Also check if any of K answers is correct (oracle)
        any_correct = any(check_correct(a, gold) for a in answers)
        
        result = {
            "question_id": q_id,
            "template": template,
            "gold": gold,
            "difficulty": difficulty,
            "dlm_answers": answers,
            "dlm_confidences": confidences,
            "majority_answer": majority_answer,
            "is_correct": is_correct,
            "any_correct": any_correct,
            "sigma2_answer": sigma2_answer,
            "sigma2_token": sigma2_token,
            "mean_confidence": statistics.mean(confidences),
            "n_unique_answers": len(set(normalize_answer(a) for a in answers)),
        }
        results.append(result)
        
        status = "✓" if is_correct else "✗"
        print(f" → {majority_answer!r} (gold: {gold!r}) {status} | σ²_ans={sigma2_answer:.3f} σ²_tok={sigma2_token:.4f}")
    
    # Save results
    results_path = CONFIG["OUTPUT_DIR"] / CONFIG["RESULTS_FILE"]
    with open(results_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"\nResults saved: {results_path}")
    
    return results


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze(results: list[dict]) -> dict:
    """Compute AUROC, ECE, Pearson ρ, and knowledge boundary metrics."""
    
    n = len(results)
    correct = [r["is_correct"] for r in results]
    incorrect = [not c for c in correct]
    sigma2_ans = [r["sigma2_answer"] for r in results]
    sigma2_tok = [r["sigma2_token"] for r in results]
    difficulties = [r["difficulty"] for r in results]
    mean_confs = [r["mean_confidence"] for r in results]
    
    accuracy = sum(correct) / n
    
    def auroc(scores: list[float], labels: list[bool]) -> float:
        """Compute AUROC for binary classification. labels=True means positive (incorrect)."""
        pairs = [(s, int(l)) for s, l in zip(scores, labels)]
        pos = [(s, l) for s, l in pairs if l == 1]
        neg = [(s, l) for s, l in pairs if l == 0]
        if not pos or not neg:
            return float('nan')
        concordant = sum(
            1 for (sp, _) in pos for (sn, _) in neg if sp > sn
        )
        tied = sum(
            0.5 for (sp, _) in pos for (sn, _) in neg if sp == sn
        )
        return (concordant + tied) / (len(pos) * len(neg))
    
    def pearson_r(x: list[float], y: list[float]) -> float:
        n = len(x)
        if n < 2:
            return float('nan')
        mx, my = sum(x)/n, sum(y)/n
        cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
        sx = math.sqrt(sum((xi - mx)**2 for xi in x) / (n - 1))
        sy = math.sqrt(sum((yi - my)**2 for yi in y) / (n - 1))
        if sx < 1e-10 or sy < 1e-10:
            return float('nan')
        return cov / (sx * sy)
    
    def ece(scores: list[float], labels: list[bool], n_bins: int = 10) -> float:
        """Expected Calibration Error. confidence = 1 - score."""
        n = len(scores)
        bins = [[] for _ in range(n_bins)]
        for s, c in zip(scores, labels):
            conf = 1 - s  # confidence = 1 - uncertainty
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            bins[bin_idx].append((conf, int(c)))
        ece_val = 0.0
        for b in bins:
            if not b:
                continue
            avg_conf = sum(x[0] for x in b) / len(b)
            avg_acc = sum(x[1] for x in b) / len(b)  # fraction correct
            ece_val += len(b) / n * abs(avg_conf - avg_acc)
        return ece_val
    
    # AUROC: does higher σ² predict incorrectness?
    auroc_ans = auroc(sigma2_ans, incorrect)
    auroc_tok = auroc(sigma2_tok, incorrect)
    auroc_conf = auroc([1 - c for c in mean_confs], incorrect)  # higher uncertainty = lower mean conf
    
    # ECE
    ece_ans = ece(sigma2_ans, correct)
    ece_tok = ece(sigma2_tok, correct)
    
    # Pearson ρ: σ² vs difficulty
    rho_ans_diff = pearson_r(sigma2_ans, difficulties)
    rho_tok_diff = pearson_r(sigma2_tok, difficulties)
    rho_conf_diff = pearson_r(mean_confs, [-d for d in difficulties])
    
    # Knowledge boundary: split into easy/medium/hard
    easy = [r for r in results if r["difficulty"] < 0.3]
    medium = [r for r in results if 0.3 <= r["difficulty"] < 0.6]
    hard = [r for r in results if r["difficulty"] >= 0.6]
    
    def group_stats(group):
        if not group:
            return {}
        return {
            "n": len(group),
            "accuracy": sum(r["is_correct"] for r in group) / len(group),
            "mean_sigma2_answer": statistics.mean(r["sigma2_answer"] for r in group),
            "mean_sigma2_token": statistics.mean(r["sigma2_token"] for r in group),
            "mean_confidence": statistics.mean(r["mean_confidence"] for r in group),
        }
    
    analysis = {
        "n_questions": n,
        "n_correct": sum(correct),
        "accuracy": accuracy,
        # AUROC (main metric — should be > 0.5 if σ² predicts error)
        "auroc_sigma2_answer": auroc_ans,
        "auroc_sigma2_token": auroc_tok,
        "auroc_mean_confidence_inv": auroc_conf,
        # ECE
        "ece_sigma2_answer": ece_ans,
        "ece_sigma2_token": ece_tok,
        # Pearson ρ (σ² vs. difficulty)
        "pearson_rho_sigma2_answer_vs_difficulty": rho_ans_diff,
        "pearson_rho_sigma2_token_vs_difficulty": rho_tok_diff,
        "pearson_rho_mean_conf_vs_neg_difficulty": rho_conf_diff,
        # Knowledge boundary decomposition
        "by_difficulty": {
            "easy": group_stats(easy),
            "medium": group_stats(medium),
            "hard": group_stats(hard),
        },
        # Sample extremes
        "most_uncertain_q": max(results, key=lambda r: r["sigma2_answer"])["template"],
        "most_certain_q": min(results, key=lambda r: r["sigma2_answer"])["template"],
    }
    
    return analysis


def print_analysis(a: dict):
    print("\n" + "="*70)
    print("BPFC PILOT RESULTS (BERT-base CPU Proxy)")
    print("="*70)
    print(f"N={a['n_questions']} questions | K={CONFIG['K_PASSES']} passes | BERT-base")
    print(f"Accuracy: {a['accuracy']:.1%} ({a['n_correct']}/{a['n_questions']} correct)")
    print()
    print("── Calibration Metrics ──────────────────────────────────────────")
    print(f"AUROC(σ²_answer → error):     {a['auroc_sigma2_answer']:.4f}  [> 0.5 = better than chance]")
    print(f"AUROC(σ²_token  → error):     {a['auroc_sigma2_token']:.4f}")
    print(f"AUROC(1−mean_conf → error):   {a['auroc_mean_confidence_inv']:.4f}")
    print(f"ECE(σ²_answer):               {a['ece_sigma2_answer']:.4f}")
    print(f"ECE(σ²_token):                {a['ece_sigma2_token']:.4f}")
    print()
    print("── Knowledge Boundary Correlation ──────────────────────────────")
    print(f"Pearson ρ(σ²_answer, difficulty): {a['pearson_rho_sigma2_answer_vs_difficulty']:.4f}")
    print(f"Pearson ρ(σ²_token, difficulty):  {a['pearson_rho_sigma2_token_vs_difficulty']:.4f}")
    print(f"Pearson ρ(mean_conf, -difficulty):{a['pearson_rho_mean_conf_vs_neg_difficulty']:.4f}")
    print()
    print("── Knowledge Decomposition ──────────────────────────────────────")
    for diff, group in a["by_difficulty"].items():
        if group:
            print(f"  {diff:8s}: acc={group['accuracy']:.0%} | σ²_ans={group['mean_sigma2_answer']:.3f} | "
                  f"σ²_tok={group['mean_sigma2_token']:.4f} | mean_conf={group['mean_confidence']:.3f}")
    print()
    print(f"Most uncertain: {a['most_uncertain_q'][:60]}")
    print(f"Most certain:   {a['most_certain_q'][:60]}")
    print("="*70)


if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv
    
    if dry_run:
        print("DRY RUN: checking imports and data...")
        from transformers import pipeline
        print(f"✓ transformers available, {len(FACTUAL_QA)} questions loaded")
        sys.exit(0)
    
    print("BPFC CPU Pilot — BERT fill-mask proxy experiment")
    print(f"N={CONFIG['N_QUESTIONS']} questions, K={CONFIG['K_PASSES']} passes")
    print("Running on CPU (BERT-base, ~110M params)")
    print()
    
    t0 = time.time()
    results = run_experiment()
    
    analysis = analyze(results)
    print_analysis(analysis)
    
    analysis_path = CONFIG["OUTPUT_DIR"] / CONFIG["ANALYSIS_FILE"]
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved: {analysis_path}")
    print(f"Total time: {time.time()-t0:.1f}s")
