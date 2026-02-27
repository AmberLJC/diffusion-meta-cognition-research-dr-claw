#!/usr/bin/env python3
"""
BPFC Statistical Analysis Module
==================================
Pure Python stdlib implementation — no numpy, scipy, or sklearn required.
Implements all metrics needed for the BPFC pilot experiment analysis.

Metrics implemented:
  - sigma2_answer: answer-level variance (Mode A)
  - sigma2_span:   token-level span variance (Mode B, requires DenoiseViz)
  - AUROC:         via Mann-Whitney U statistic (exact, no dependencies)
  - ECE:           Expected Calibration Error (B=10 bins)
  - Pearson ρ:     Pearson correlation coefficient
  - Bootstrap CI:  Confidence intervals via bootstrap resampling
  - KnowledgeDecomp: 4-quadrant "Known/Lucky/Mistake/Unknown" table

Author: Dr. Claw | 2026-02-27
"""

import math
import random
import json
import string
import re
from typing import List, Dict, Optional, Tuple, Any


# ─── Basic Statistics ──────────────────────────────────────────────────────────

def mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def variance(xs: List[float], ddof: int = 1) -> float:
    """Sample variance with Bessel's correction (ddof=1) or population (ddof=0)."""
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    ss = sum((x - mu) ** 2 for x in xs)
    return ss / (len(xs) - ddof)


def std(xs: List[float], ddof: int = 1) -> float:
    return math.sqrt(variance(xs, ddof))


def median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return float(s[n // 2])


def percentile(xs: List[float], p: float) -> float:
    """Linear interpolation percentile, p in [0, 100]."""
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    idx = (n - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


# ─── Text Normalization (TriviaQA protocol) ───────────────────────────────────

def normalize_answer(text: str) -> str:
    """
    Normalize answer for lexical comparison (TriviaQA evaluation protocol).
    - Lowercase
    - Strip punctuation
    - Normalize whitespace
    - Remove common articles (a, an, the)
    """
    text = text.lower().strip()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove leading/trailing articles
    text = re.sub(r'^(a|an|the)\s+', '', text)
    text = re.sub(r'\s+(a|an|the)$', '', text)
    return text


def exact_match(pred: str, gold_list: List[str]) -> bool:
    """TriviaQA-style exact match: pred matches any gold answer after normalization."""
    pred_norm = normalize_answer(pred)
    return any(normalize_answer(g) == pred_norm for g in gold_list)


def contains_match(pred: str, gold_list: List[str]) -> bool:
    """Relaxed: pred contains any gold answer (handles verbose outputs)."""
    pred_norm = normalize_answer(pred)
    return any(normalize_answer(g) in pred_norm for g in gold_list)


# ─── Sigma² Metrics ───────────────────────────────────────────────────────────

def sigma2_answer(answers: List[str]) -> float:
    """
    Mode A: Answer-level variance.
    σ²_answer = 1 - mean_pairwise_agreement
    Range [0, 1]: 0 = all identical, 1 = all different.
    """
    K = len(answers)
    if K < 2:
        return 0.0
    norm = [normalize_answer(a) for a in answers]
    agree_count = 0
    pair_count = 0
    for i in range(K):
        for j in range(i + 1, K):
            pair_count += 1
            if norm[i] == norm[j]:
                agree_count += 1
    if pair_count == 0:
        return 0.0
    return 1.0 - (agree_count / pair_count)


def sigma2_span(token_confidence_matrix: List[List[float]], answer_span_indices: Optional[List[int]] = None) -> float:
    """
    Mode B: Token-level span variance.
    
    Args:
        token_confidence_matrix: K × L matrix, where [k][i] = confidence of token i in pass k
        answer_span_indices: Indices of answer tokens. If None, uses all positions.
    
    Returns:
        σ²_span = mean over answer span of Var_k[c_i^(k)]
    """
    K = len(token_confidence_matrix)
    if K < 2:
        return 0.0
    
    if not token_confidence_matrix[0]:
        return 0.0
    
    L = len(token_confidence_matrix[0])
    if answer_span_indices is None:
        answer_span_indices = list(range(L))
    
    if not answer_span_indices:
        return 0.0
    
    span_variances = []
    for i in answer_span_indices:
        confs_i = [token_confidence_matrix[k][i] for k in range(K) if i < len(token_confidence_matrix[k])]
        if len(confs_i) >= 2:
            span_variances.append(variance(confs_i, ddof=1))
    
    return mean(span_variances) if span_variances else 0.0


def sigma2_span_from_denoiseviz(
    denoiseviz_outputs: List[List[Dict]],
    prompt_len: int
) -> Tuple[float, List[float]]:
    """
    Compute σ²_span from K DenoiseViz outputs.
    
    Args:
        denoiseviz_outputs: K lists of {token: str, class_or_confidence: float|str|None}
        prompt_len: Number of tokens in the prompt (to identify answer span)
    
    Returns:
        (sigma2_span_value, per_token_variances)
    """
    K = len(denoiseviz_outputs)
    if K < 2 or not denoiseviz_outputs[0]:
        return 0.0, []
    
    # Extract numeric confidence values (skip prompt tokens)
    def extract_confidence(tok_dict: Dict) -> Optional[float]:
        val = tok_dict.get('class_or_confidence')
        if isinstance(val, (int, float)):
            return float(val)
        if val is None or val == 'mask':
            return None  # Still masked at end of denoising — unusual
        return None
    
    # Find answer span (all positions after prompt_len)
    L = len(denoiseviz_outputs[0])
    answer_span = [i for i in range(prompt_len, L)]
    
    if not answer_span:
        return 0.0, []
    
    # Build confidence matrix for answer span
    conf_matrix = []
    for k in range(K):
        row = []
        for i in answer_span:
            if i < len(denoiseviz_outputs[k]):
                c = extract_confidence(denoiseviz_outputs[k][i])
                row.append(c if c is not None else 0.5)  # impute 0.5 for missing
            else:
                row.append(0.5)
        conf_matrix.append(row)
    
    # Compute per-position variance
    per_token_vars = []
    for j in range(len(answer_span)):
        confs_j = [conf_matrix[k][j] for k in range(K)]
        per_token_vars.append(variance(confs_j, ddof=1))
    
    return mean(per_token_vars), per_token_vars


# ─── AUROC via Mann-Whitney U ─────────────────────────────────────────────────

def auroc(scores: List[float], labels: List[int]) -> float:
    """
    Compute AUROC via Mann-Whitney U statistic.
    labels: 1 = positive (incorrect/uncertain), 0 = negative (correct/certain)
    scores: higher score = model predicts positive
    
    AUROC = P(score_pos > score_neg)
    """
    positives = [s for s, l in zip(scores, labels) if l == 1]
    negatives = [s for s, l in zip(scores, labels) if l == 0]
    
    if not positives or not negatives:
        return 0.5  # degenerate case
    
    # Mann-Whitney U: count pairs where pos > neg
    concordant = sum(1 for p in positives for n in negatives if p > n)
    tied = sum(0.5 for p in positives for n in negatives if p == n)
    total = len(positives) * len(negatives)
    
    return (concordant + tied) / total


def roc_curve(scores: List[float], labels: List[int]) -> Tuple[List[float], List[float]]:
    """Compute ROC curve (FPR, TPR) for plotting."""
    # Sort by descending score
    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return [0.0, 1.0], [0.0, 1.0]
    
    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0
    
    for s, l in pairs:
        if l == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    
    return fpr_list, tpr_list


# ─── Expected Calibration Error ───────────────────────────────────────────────

def expected_calibration_error(
    uncertainties: List[float],
    corrects: List[int],
    n_bins: int = 10
) -> float:
    """
    ECE = Σ_b (|B_b|/N) |acc(B_b) - (1 - mean_uncertainty(B_b))|
    
    Args:
        uncertainties: σ²_span values per question (0 = certain, 1 = uncertain)
        corrects: 1 = correct, 0 = incorrect
        n_bins: Number of equal-frequency bins
    """
    N = len(uncertainties)
    if N == 0:
        return 0.0
    
    # Sort by uncertainty
    sorted_pairs = sorted(zip(uncertainties, corrects))
    
    # Split into n_bins equal-frequency bins
    ece = 0.0
    bin_size = N / n_bins
    
    for b in range(n_bins):
        start = int(b * bin_size)
        end = int((b + 1) * bin_size)
        if end > N:
            end = N
        if start >= end:
            continue
        
        bin_u = [sorted_pairs[i][0] for i in range(start, end)]
        bin_c = [sorted_pairs[i][1] for i in range(start, end)]
        
        bin_acc = mean(bin_c)
        bin_conf = 1.0 - mean(bin_u)  # Confidence = 1 - uncertainty
        
        ece += (len(bin_u) / N) * abs(bin_acc - bin_conf)
    
    return ece


# ─── Pearson Correlation ──────────────────────────────────────────────────────

def pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return 0.0
    
    mx, my = mean(xs), mean(ys)
    
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    
    if sx == 0 or sy == 0:
        return 0.0
    
    return cov / (sx * sy)


def spearman_r(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation."""
    n = len(xs)
    if n < 2:
        return 0.0
    
    def rank_list(vals):
        sorted_idx = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and vals[sorted_idx[j]] == vals[sorted_idx[i]]:
                j += 1
            avg_rank = (i + j - 1) / 2
            for k in range(i, j):
                ranks[sorted_idx[k]] = avg_rank + 1  # 1-indexed
            i = j
        return ranks
    
    rx, ry = rank_list(xs), rank_list(ys)
    return pearson_r(rx, ry)


# ─── Bootstrap Confidence Intervals ───────────────────────────────────────────

def bootstrap_ci(
    func,
    data: List[Any],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for a scalar statistic.
    
    Returns: (point_estimate, lower_ci, upper_ci)
    """
    rng = random.Random(seed)
    n = len(data)
    
    point_est = func(data)
    
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        try:
            boot_stats.append(func(sample))
        except Exception:
            pass
    
    if not boot_stats:
        return point_est, point_est, point_est
    
    lo = percentile(boot_stats, 100 * alpha / 2)
    hi = percentile(boot_stats, 100 * (1 - alpha / 2))
    return point_est, lo, hi


# ─── Knowledge Decomposition ──────────────────────────────────────────────────

def knowledge_decomposition(
    uncertainties: List[float],
    corrects: List[int],
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Four-quadrant knowledge decomposition.
    
    Quadrants:
      Known:           low σ², correct
      Lucky Guess:     high σ², correct  ← invisible to accuracy-only eval
      Confident Mistake: low σ², incorrect
      Unknown:         high σ², incorrect
    
    Args:
        uncertainties: σ²_span per question
        corrects: 1 = correct answer, 0 = incorrect
        threshold: σ² threshold; if None, uses median
    """
    N = len(uncertainties)
    if threshold is None:
        threshold = median(uncertainties)
    
    counts = {"known": 0, "lucky_guess": 0, "confident_mistake": 0, "unknown": 0}
    questions_by_quadrant = {"known": [], "lucky_guess": [], "confident_mistake": [], "unknown": []}
    
    for i, (u, c) in enumerate(zip(uncertainties, corrects)):
        low_uncertainty = u <= threshold
        correct = c == 1
        
        if low_uncertainty and correct:
            q = "known"
        elif not low_uncertainty and correct:
            q = "lucky_guess"
        elif low_uncertainty and not correct:
            q = "confident_mistake"
        else:
            q = "unknown"
        
        counts[q] += 1
        questions_by_quadrant[q].append(i)
    
    fractions = {k: v / N for k, v in counts.items()}
    
    return {
        "threshold": threshold,
        "counts": counts,
        "fractions": fractions,
        "questions_by_quadrant": questions_by_quadrant,
        "N": N,
        # Derived metrics
        "lucky_guess_rate": fractions["lucky_guess"],  # Key metric: acc-invisible uncertainty
        "confident_mistake_rate": fractions["confident_mistake"],  # Hard failure mode
        "known_fraction": fractions["known"],
        "unknown_fraction": fractions["unknown"],
    }


# ─── Full Analysis Pipeline ───────────────────────────────────────────────────

def run_analysis(
    results: List[Dict],
    entity_frequencies: Optional[Dict[str, float]] = None,
    n_bootstrap: int = 1000
) -> Dict[str, Any]:
    """
    Run full BPFC analysis pipeline.
    
    Args:
        results: List of per-question result dicts with fields:
            - question: str
            - gold_answers: List[str]
            - dlm_answers: List[str]  (K answers from LLaDA)
            - token_confidences: Optional[List[List[float]]]  (K × L matrix, Mode B)
            - answer_span_indices: Optional[List[int]]
            - entity_name: Optional[str]
        entity_frequencies: {entity_name: freq_value}  (for RQ3)
        n_bootstrap: Bootstrap resamples for CIs
    
    Returns:
        Complete analysis dict with all metrics
    """
    N = len(results)
    if N == 0:
        return {"error": "No results to analyze"}
    
    # 1. Compute per-question correctness
    corrects = []
    for r in results:
        # Use majority vote as "correct" label
        majority = max(set(normalize_answer(a) for a in r['dlm_answers']),
                      key=lambda x: sum(1 for a in r['dlm_answers'] if normalize_answer(a) == x))
        is_correct = exact_match(majority, r['gold_answers']) or contains_match(majority, r['gold_answers'])
        corrects.append(1 if is_correct else 0)
    
    accuracy = mean(corrects)
    
    # 2. Mode A: σ²_answer
    sig2_answers = [sigma2_answer(r['dlm_answers']) for r in results]
    
    # 3. Mode B: σ²_span (if token_confidences available)
    sig2_spans = []
    mode_b_available = all('token_confidences' in r and r['token_confidences'] for r in results)
    
    if mode_b_available:
        for r in results:
            confs = r['token_confidences']
            span_idx = r.get('answer_span_indices')
            val = sigma2_span(confs, span_idx)
            sig2_spans.append(val)
    
    # 4. AUROC
    errors = [1 - c for c in corrects]  # 1 = incorrect, 0 = correct
    
    auroc_mode_a = auroc(sig2_answers, errors)
    auroc_mode_b = auroc(sig2_spans, errors) if sig2_spans else None
    
    # 5. Bootstrap CIs for AUROC
    def auroc_from_data(data):
        scores_sub = [d[0] for d in data]
        labels_sub = [d[1] for d in data]
        return auroc(scores_sub, labels_sub)
    
    paired_a = list(zip(sig2_answers, errors))
    _, auroc_a_lo, auroc_a_hi = bootstrap_ci(auroc_from_data, paired_a, n_bootstrap)
    
    # 6. ECE
    ece_mode_a = expected_calibration_error(sig2_answers, corrects)
    ece_mode_b = expected_calibration_error(sig2_spans, corrects) if sig2_spans else None
    
    # 7. Knowledge decomposition
    decomp = knowledge_decomposition(sig2_answers, corrects)
    
    # 8. Entity frequency correlation (RQ3)
    freq_correlation = None
    if entity_frequencies:
        freq_vals = []
        sig2_for_freq = []
        for r, s2 in zip(results, sig2_answers):
            entity = r.get('entity_name')
            if entity and entity in entity_frequencies:
                freq_vals.append(-math.log10(max(entity_frequencies[entity], 1)))
                sig2_for_freq.append(s2)
        
        if len(freq_vals) >= 5:
            freq_correlation = {
                "pearson_r": pearson_r(sig2_for_freq, freq_vals),
                "spearman_r": spearman_r(sig2_for_freq, freq_vals),
                "n_with_freq": len(freq_vals)
            }
    
    # 9. Summary statistics for σ²
    def summarize(vals):
        return {
            "mean": mean(vals),
            "std": std(vals),
            "median": median(vals),
            "min": min(vals),
            "max": max(vals),
            "p25": percentile(vals, 25),
            "p75": percentile(vals, 75),
        }
    
    analysis = {
        "N": N,
        "accuracy": accuracy,
        "n_correct": sum(corrects),
        "n_incorrect": N - sum(corrects),
        
        "mode_a": {
            "auroc": auroc_mode_a,
            "auroc_ci_95": (auroc_a_lo, auroc_a_hi),
            "ece": ece_mode_a,
            "sigma2_stats": summarize(sig2_answers),
        },
        
        "mode_b": {
            "available": mode_b_available,
            "auroc": auroc_mode_b,
            "ece": ece_mode_b,
            "sigma2_stats": summarize(sig2_spans) if sig2_spans else None,
        } if mode_b_available or sig2_spans else {"available": False},
        
        "knowledge_decomposition": decomp,
        "entity_frequency_correlation": freq_correlation,
        
        # Hypothesis tests
        "h1_supported": auroc_mode_a > 0.55,  # AUROC > 0.55
        "h2_supported": (auroc_mode_b is not None and auroc_mode_b > auroc_mode_a),
        "h3_supported": (freq_correlation and freq_correlation["pearson_r"] > 0.25),
    }
    
    return analysis


# ─── Mock Data Dry Run ────────────────────────────────────────────────────────

def generate_mock_results(N: int = 50, K: int = 8, seed: int = 42) -> List[Dict]:
    """
    Generate synthetic pilot results to validate the analysis pipeline.
    
    Stratified by difficulty (N//3 each tier):
      Easy   (high freq): p_correct=0.9/pass → majority always correct, low σ²
      Medium (mid freq):  p_correct=0.6/pass → majority ~70% correct, medium σ²
      Hard   (low freq):  p_correct=0.2/pass → majority ~20% correct, high σ²
    
    Expected AUROC ~ 0.68-0.72 (high σ² predicts incorrectness).
    """
    rng = random.Random(seed)
    
    EASY_FACTS = [
        ("Capital of France?", "Paris", 1e7),
        ("Capital of Germany?", "Berlin", 8e6),
        ("Author of Hamlet?", "Shakespeare", 5e6),
        ("Largest planet?", "Jupiter", 4e6),
        ("Chemical formula of water?", "H2O", 2e7),
        ("Capital of Japan?", "Tokyo", 9e6),
    ]
    MEDIUM_FACTS = [
        ("Who discovered penicillin?", "Fleming", 5e5),
        ("Year WWI ended?", "1918", 3e5),
        ("Who invented the telephone?", "Bell", 8e5),
        ("Capital of Australia?", "Canberra", 2e5),
        ("Element with symbol Fe?", "Iron", 6e5),
        ("Author of Don Quixote?", "Cervantes", 4e5),
    ]
    HARD_FACTS = [
        ("Who wrote 'The Tin Drum'?", "Grass", 3e4),
        ("Capital of Kyrgyzstan?", "Bishkek", 2e4),
        ("Year Ottoman Empire founded?", "1299", 1e4),
        ("Smallest country by area?", "Vatican", 5e4),
        ("Who invented the printing press?", "Gutenberg", 8e4),
        ("Language spoken in Suriname?", "Dutch", 2e4),
    ]
    
    all_answers = (
        [f[1] for f in EASY_FACTS] +
        [f[1] for f in MEDIUM_FACTS] +
        [f[1] for f in HARD_FACTS]
    )
    
    # Build stratified sample
    n_easy = N // 3
    n_medium = N // 3
    n_hard = N - n_easy - n_medium
    
    tiers = (
        [(q, g, freq, "easy", 0.9, (0.75, 0.99)) for q, g, freq in EASY_FACTS] * (n_easy // len(EASY_FACTS) + 1)
    )[:n_easy] + (
        [(q, g, freq, "medium", 0.6, (0.4, 0.95)) for q, g, freq in MEDIUM_FACTS] * (n_medium // len(MEDIUM_FACTS) + 1)
    )[:n_medium] + (
        [(q, g, freq, "hard", 0.2, (0.05, 0.7)) for q, g, freq in HARD_FACTS] * (n_hard // len(HARD_FACTS) + 1)
    )[:n_hard]
    
    rng.shuffle(tiers)
    
    results = []
    for i, (q, gold, freq, difficulty, p_correct, conf_range) in enumerate(tiers):
        
        # Generate K answers
        answers = []
        for k in range(K):
            if rng.random() < p_correct:
                answers.append(gold)
            else:
                wrong = rng.choice([a for a in all_answers if a != gold])
                answers.append(wrong)
        
        # Generate mock DenoiseViz token confidences
        L_prompt = 10  # prompt tokens (fixed)
        L_answer = 3   # answer tokens
        L_total = L_prompt + L_answer
        conf_lo, conf_hi = conf_range
        
        token_confs = []
        for k in range(K):
            confs = []
            for j in range(L_total):
                if j < L_prompt:
                    confs.append(rng.uniform(0.85, 1.0))  # prompt always confident
                else:
                    # Answer tokens: span confidence matches difficulty
                    c = rng.uniform(conf_lo, conf_hi)
                    confs.append(c)
            token_confs.append(confs)
        
        results.append({
            "question_id": f"mock_{i:03d}",
            "question": q,
            "gold_answers": [gold],
            "dlm_answers": answers,
            "token_confidences": token_confs,
            "answer_span_indices": list(range(L_prompt, L_total)),
            "entity_name": gold,
            "entity_frequency": freq,
            "difficulty": difficulty,
        })
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("BPFC Statistical Analysis — Dry Run with Mock Data")
    print("=" * 60)
    
    # Generate mock results
    results = generate_mock_results(N=50, K=8, seed=42)
    entity_frequencies = {r['entity_name']: r['entity_frequency'] for r in results}
    
    # Run full analysis
    analysis = run_analysis(results, entity_frequencies=entity_frequencies, n_bootstrap=500)
    
    print(f"\n📊 Primary Results (N={analysis['N']}, K=8)")
    print(f"   Accuracy:        {analysis['accuracy']:.1%}")
    
    print(f"\n🎯 Mode A (Answer-Level Variance, σ²_answer):")
    ma = analysis['mode_a']
    print(f"   AUROC:           {ma['auroc']:.3f} "
          f"(95% CI: {ma['auroc_ci_95'][0]:.3f}–{ma['auroc_ci_95'][1]:.3f})")
    print(f"   ECE:             {ma['ece']:.3f}")
    print(f"   σ² mean±std:     {ma['sigma2_stats']['mean']:.3f} ± {ma['sigma2_stats']['std']:.3f}")
    
    if analysis['mode_b'].get('available'):
        mb = analysis['mode_b']
        print(f"\n🎯 Mode B (Token-Level Variance, σ²_span):")
        print(f"   AUROC:           {mb['auroc']:.3f}")
        print(f"   ECE:             {mb['ece']:.3f}")
    
    print(f"\n📚 Knowledge Decomposition (median σ² split):")
    kd = analysis['knowledge_decomposition']
    for k, frac in kd['fractions'].items():
        print(f"   {k:25s}: {kd['counts'][k]:3d} ({frac:.1%})")
    
    if analysis.get('entity_frequency_correlation'):
        fc = analysis['entity_frequency_correlation']
        print(f"\n🔗 Entity Frequency Correlation (n={fc['n_with_freq']}):")
        print(f"   Pearson ρ:       {fc['pearson_r']:.3f}")
        print(f"   Spearman ρ:      {fc['spearman_r']:.3f}")
    
    print(f"\n✅ Hypothesis Tests:")
    print(f"   H1 (AUROC > 0.55): {'SUPPORTED' if analysis['h1_supported'] else 'NOT SUPPORTED'}")
    print(f"   H2 (B > A):        {'SUPPORTED' if analysis['h2_supported'] else 'NOT SUPPORTED'}")
    print(f"   H3 (ρ_freq > 0.25):{'SUPPORTED' if analysis['h3_supported'] else 'NOT SUPPORTED'}")
    
    # Validate the pipeline
    print(f"\n✓ Pipeline validation: all metrics computed without numpy/scipy")
    print(f"  Ready for real data from bpfc_pilot.py experiment run")
    
    # Save to file for inspection
    import os
    os.makedirs("../data", exist_ok=True)
    with open("../data/dry_run_analysis.json", "w") as f:
        # Make serializable
        def make_serial(obj):
            if isinstance(obj, dict):
                return {k: make_serial(v) for k, v in obj.items() 
                        if k != 'questions_by_quadrant'}  # Skip verbose lists
            elif isinstance(obj, list):
                return [make_serial(x) for x in obj]
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, (int, float)):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            return obj
        json.dump(make_serial(analysis), f, indent=2)
    print("  Saved to data/dry_run_analysis.json")
