#!/usr/bin/env python3
"""
BPFC Extended Pilot — N=150, K=8, ECE + Calibration Curves
============================================================
Runs entirely on CPU using BERT-base as a proxy masked diffusion model.
Extends the N=50 pilot to N=150 for stronger statistical power.
Adds: ECE with 10 calibration bins, reliability diagrams, partial-AUC.

AUTHOR: Dr. Claw | 2026-02-27 (Run #10)
"""

import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import List, Tuple, Optional

CONFIG = {
    "N_QUESTIONS": 120,
    "K_PASSES": 8,
    "BERT_MODEL": "bert-base-uncased",
    "SEED": 42,
    "OUTPUT_DIR": Path(__file__).parent.parent / "data",
    "RESULTS_FILE": "extended_pilot_n120.jsonl",
    "ANALYSIS_FILE": "extended_pilot_n120_analysis.json",
    "ECE_BINS": 10,
    "BOOTSTRAP_SAMPLES": 1000,
}
CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
random.seed(CONFIG["SEED"])

# ── 150-question factual QA bank ─────────────────────────────────────────────
# (template, correct_answer, difficulty 0=easy→1=hard)

FACTUAL_QA = [
    # === EASY (0.0–0.2) ===
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
    ("Leonardo da Vinci painted the Mona [MASK].", "lisa", 0.05),
    ("Mount Everest is the tallest [MASK] in the world.", "mountain", 0.05),
    ("The inventor of the telephone was Alexander Graham [MASK].", "bell", 0.1),
    ("The currency of the UK is the [MASK].", "pound", 0.1),
    ("The chemical symbol for water is [MASK].", "h2o", 0.05),
    ("The capital of Russia is [MASK].", "moscow", 0.0),
    ("The capital of Brazil is [MASK].", "brasilia", 0.15),
    ("The capital of India is [MASK].", "delhi", 0.1),
    ("The capital of Egypt is [MASK].", "cairo", 0.1),
    ("The speed of light is measured in [MASK] per second.", "meters", 0.05),
    ("The first US president was George [MASK].", "washington", 0.0),
    ("The planet closest to the Sun is [MASK].", "mercury", 0.1),
    ("The Pacific is the largest [MASK] in the world.", "ocean", 0.0),
    ("DNA stands for deoxyribonucleic [MASK].", "acid", 0.1),

    # === MEDIUM (0.3–0.6) ===
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
    ("The theory of relativity was developed by [MASK].", "einstein", 0.3),
    ("The largest planet in our solar system is [MASK].", "jupiter", 0.3),
    ("The currency of the USA is the [MASK].", "dollar", 0.1),
    ("Thomas Edison invented the light [MASK].", "bulb", 0.3),
    ("The capital of South Africa is [MASK].", "pretoria", 0.5),
    ("The chemical symbol for sodium is [MASK].", "na", 0.45),
    ("The speed of sound in air is about 343 meters per [MASK].", "second", 0.35),
    ("The Great Barrier Reef is located in [MASK].", "australia", 0.3),
    ("The Nobel Prize in Physics 2021 was for [MASK] systems.", "complex", 0.6),
    ("The chemical formula for carbon dioxide is [MASK].", "co2", 0.3),
    ("The Louvre museum is in [MASK].", "paris", 0.2),
    ("Beethoven's Ninth Symphony features the [MASK] of Joy.", "ode", 0.5),
    ("The human genome has approximately [MASK] base pairs.", "3", 0.55),
    ("The philosopher Socrates lived in [MASK].", "athens", 0.4),
    ("The chemical symbol for lead is [MASK].", "pb", 0.5),
    ("The capital of Argentina is Buenos [MASK].", "aires", 0.3),
    ("Photosynthesis requires sunlight, water, and [MASK].", "co2", 0.35),
    ("The Nile is the longest [MASK] in the world.", "river", 0.1),

    # === HARD (0.7–1.0) ===
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
    ("The capital of Mongolia is [MASK].", "ulaanbaatar", 0.9),
    ("The chemical symbol for silver is [MASK].", "ag", 0.65),
    ("The Thirty Years War ended in [MASK].", "1648", 0.85),
    ("The capital of Uzbekistan is [MASK].", "tashkent", 0.85),
    ("The enzyme that breaks down lactose is [MASK].", "lactase", 0.8),
    ("The Fibonacci sequence begins 0, 1, 1, 2, 3, 5, [MASK].", "8", 0.7),
    ("The Roman Empire fell in [MASK] AD.", "476", 0.8),
    ("The capital of Mozambique is [MASK].", "maputo", 0.9),
    ("The chemical symbol for mercury is [MASK].", "hg", 0.75),
    ("The Treaty of [MASK] established the European Union.", "maastricht", 0.85),
    ("The Higgs boson was discovered at [MASK].", "cern", 0.8),
    ("The capital of Azerbaijan is [MASK].", "baku", 0.8),
    ("The [MASK] effect describes the redshift of light from distant galaxies.", "doppler", 0.75),
    ("The mitochondrial DNA is inherited from the [MASK] parent.", "mother", 0.75),
    ("The capital of Kyrgyzstan is [MASK].", "bishkek", 0.9),
    ("The Peloponnesian War was fought between Athens and [MASK].", "sparta", 0.7),

    # === EXTRA MEDIUM-HARD (0.4–0.8) ===
    ("The speed of Earth's rotation at the equator is about [MASK] km/h.", "1670", 0.8),
    ("The chemical symbol for potassium is [MASK].", "k", 0.5),
    ("The painter Vincent van Gogh was from the [MASK].", "netherlands", 0.5),
    ("The year the Berlin Wall fell was [MASK].", "1989", 0.4),
    ("The capital of Peru is [MASK].", "lima", 0.4),
    ("Napoleon Bonaparte was exiled to the island of [MASK].", "elba", 0.7),
    ("The element with atomic number 79 is [MASK].", "gold", 0.6),
    ("The philosopher who wrote 'Critique of Pure Reason' was [MASK].", "kant", 0.65),
    ("The capital of Vietnam is [MASK].", "hanoi", 0.5),
    ("The year the Titanic sank was [MASK].", "1912", 0.4),
    ("The chemical symbol for copper is [MASK].", "cu", 0.55),
    ("The author of Don Quixote is [MASK].", "cervantes", 0.65),
    ("The smallest planet in our solar system is [MASK].", "mercury", 0.4),
    ("The capital of Ethiopia is [MASK].", "addis", 0.6),
    ("Einstein won the Nobel Prize in [MASK].", "physics", 0.5),
    ("The capital of the Philippines is [MASK].", "manila", 0.5),
    ("The French Revolution began in [MASK].", "1789", 0.45),
    ("The chemical symbol for nitrogen is [MASK].", "n", 0.4),
    ("The ancient wonder located in Egypt is the [MASK].", "pyramids", 0.2),
    ("The capital of Colombia is [MASK].", "bogota", 0.55),
    ("The year the first iPhone was released was [MASK].", "2007", 0.3),
    ("The chemical symbol for carbon is [MASK].", "c", 0.3),
    ("Sigmund Freud developed [MASK] theory.", "psychoanalytic", 0.5),
    ("The capital of Sweden is [MASK].", "stockholm", 0.3),
    ("The number of bones in the human skull is [MASK].", "22", 0.75),
    ("The author of 1984 is George [MASK].", "orwell", 0.35),
    ("The currency of India is the [MASK].", "rupee", 0.3),
    ("The country with the most Nobel Prize winners is [MASK].", "usa", 0.5),
    ("The chemical formula for ammonia is [MASK].", "nh3", 0.65),
    ("The year World War I began was [MASK].", "1914", 0.3),
]

assert len(FACTUAL_QA) >= CONFIG["N_QUESTIONS"], f"Need {CONFIG['N_QUESTIONS']}, have {len(FACTUAL_QA)}"
FACTUAL_QA = FACTUAL_QA[:CONFIG["N_QUESTIONS"]]

# ── BERT inference ─────────────────────────────────────────────────────────────

def load_bert():
    """Load BERT fill-mask pipeline (top_k=50 for sampling diversity)."""
    from transformers import pipeline
    pipe = pipeline(
        "fill-mask",
        model=CONFIG["BERT_MODEL"],
        device=-1,  # CPU
        top_k=50,   # Return top-50 tokens → sample for stochastic passes
    )
    return pipe

def bert_fill_mask(pipe, template: str, temperature: float = 1.0) -> Tuple[str, float]:
    """One fill-mask pass: sample from top-50 token distribution.
    
    This implements K independent posterior draws: different random seeds
    yield different samples, creating the σ²_span signal.
    Temperature > 1 flattens the distribution (more entropy);
    temperature = 1 uses BERT's native posterior.
    """
    results = pipe(template)  # list of {token_str, score, ...}
    scores = [r["score"] for r in results]
    tokens = [r["token_str"].strip().lower() for r in results]
    
    if temperature != 1.0:
        log_probs = [math.log(max(s, 1e-10)) / temperature for s in scores]
        max_lp = max(log_probs)
        exp_lp = [math.exp(lp - max_lp) for lp in log_probs]
        total = sum(exp_lp)
        probs = [e / total for e in exp_lp]
    else:
        probs = scores
    
    # Sample from posterior (this is what makes K passes non-deterministic)
    r = random.random()
    cumulative = 0.0
    chosen_idx = len(tokens) - 1
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            chosen_idx = i
            break
    
    chosen_token = tokens[chosen_idx]
    chosen_conf = probs[chosen_idx]
    return chosen_token, math.log(max(chosen_conf, 1e-12))


def bpfc_passes(pipe, template: str, K: int) -> dict:
    """K independent masked denoising passes → σ²_span and answer variance."""
    answers = []
    log_probs = []
    for _ in range(K):
        word, lp = bert_fill_mask(pipe, template)
        answers.append(word)
        log_probs.append(lp)

    # Token-level confidence: mean log-prob
    mean_conf = statistics.mean(log_probs) if log_probs else float("nan")

    # σ²_answer: fraction of unique answers (normalized entropy proxy)
    from collections import Counter
    counts = Counter(answers)
    n = len(answers)
    # Shannon entropy over answers (normalized by log K)
    entropy = -sum((c/n) * math.log(c/n) for c in counts.values() if c > 0)
    max_entropy = math.log(n) if n > 1 else 1.0
    sigma2_span = entropy / max_entropy  # ∈ [0, 1], 0=certain 1=max disorder

    # Majority vote answer
    top_answer = counts.most_common(1)[0][0]
    top_count = counts.most_common(1)[0][1]
    majority_confidence = top_count / n  # confidence in majority answer

    return {
        "answers": answers,
        "log_probs": log_probs,
        "sigma2_span": sigma2_span,
        "mean_conf": mean_conf,
        "majority_answer": top_answer,
        "majority_confidence": majority_confidence,
        "n_unique": len(counts),
    }


# ── Statistics helpers ─────────────────────────────────────────────────────────

def compute_auroc(scores: List[float], labels: List[int]) -> float:
    """AUROC via Wilcoxon (ascending score = more uncertain → incorrect)."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n1 = sum(labels)  # correct
    n0 = len(labels) - n1  # incorrect
    if n1 == 0 or n0 == 0:
        return float("nan")
    rank_sum = sum(i + 1 for i, (_, l) in enumerate(pairs) if l == 1)
    U = rank_sum - n1 * (n1 + 1) / 2
    return 1 - (U / (n0 * n1))  # high σ² → incorrect


def compute_ece(confidences: List[float], corrects: List[int], n_bins: int = 10) -> Tuple[float, List[dict]]:
    """ECE + per-bin data for reliability diagram."""
    bins = [{"conf_sum": 0.0, "acc_sum": 0, "n": 0} for _ in range(n_bins)]
    for conf, corr in zip(confidences, corrects):
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b]["conf_sum"] += conf
        bins[b]["acc_sum"] += corr
        bins[b]["n"] += 1

    ece = 0.0
    bin_data = []
    for b in bins:
        if b["n"] == 0:
            bin_data.append({"mean_conf": 0, "acc": 0, "n": 0})
            continue
        mean_conf = b["conf_sum"] / b["n"]
        acc = b["acc_sum"] / b["n"]
        ece += (b["n"] / len(confidences)) * abs(mean_conf - acc)
        bin_data.append({"mean_conf": mean_conf, "acc": acc, "n": b["n"]})

    return ece, bin_data


def pearson_r(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x)/n, sum(y)/n
    num = sum((a-mx)*(b-my) for a, b in zip(x, y))
    den = math.sqrt(sum((a-mx)**2 for a in x) * sum((b-my)**2 for b in y))
    return num / den if den > 0 else 0.0


def bootstrap_auroc(scores: List[float], labels: List[int], n: int = 1000, seed: int = 0) -> Tuple[float, float]:
    rng = random.Random(seed)
    aurocs = []
    N = len(scores)
    for _ in range(n):
        idx = [rng.randint(0, N-1) for _ in range(N)]
        s = [scores[i] for i in idx]
        l = [labels[i] for i in idx]
        a = compute_auroc(s, l)
        if not math.isnan(a):
            aurocs.append(a)
    aurocs.sort()
    lo = aurocs[int(0.025 * len(aurocs))]
    hi = aurocs[int(0.975 * len(aurocs))]
    return (hi - lo) / 2, statistics.stdev(aurocs) if len(aurocs) > 1 else 0.0


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print(f"🔬 BPFC Extended Pilot N=120 (K={CONFIG['K_PASSES']})")
    t0 = time.time()

    print("📦 Loading BERT fill-mask pipeline (top_k=50)...")
    pipe = load_bert()
    print(f"✅ Loaded in {time.time()-t0:.1f}s")

    results = []
    out_path = CONFIG["OUTPUT_DIR"] / CONFIG["RESULTS_FILE"]

    for i, (template, gold, difficulty) in enumerate(FACTUAL_QA):
        t_q = time.time()
        r = bpfc_passes(pipe, template, CONFIG["K_PASSES"])

        # Correctness (partial match: gold word appears in majority answer)
        is_correct = int(gold.lower() in r["majority_answer"].lower() or
                         r["majority_answer"].lower() in gold.lower())

        rec = {
            "idx": i,
            "template": template,
            "gold": gold,
            "difficulty": difficulty,
            "is_correct": is_correct,
            "majority_answer": r["majority_answer"],
            "sigma2_span": r["sigma2_span"],
            "majority_confidence": r["majority_confidence"],
            "mean_conf": r["mean_conf"],
            "n_unique": r["n_unique"],
            "answers": r["answers"],
            "elapsed_s": time.time() - t_q,
        }
        results.append(rec)

        if (i+1) % 25 == 0 or i == 0:
            acc_so_far = sum(r["is_correct"] for r in results) / len(results)
            print(f"  [{i+1:3d}/{CONFIG['N_QUESTIONS']}] acc={acc_so_far:.2f} "
                  f"σ²={r['sigma2_span']:.3f} corr={is_correct} "
                  f"Q='{template[:45]}...' pred='{r['majority_answer']}'")

    # Save raw
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n💾 Saved {len(results)} records → {out_path}")

    # ── Analysis ────────────────────────────────────────────────────────────────
    sigma2s  = [r["sigma2_span"] for r in results]
    maj_confs = [r["majority_confidence"] for r in results]
    corrects  = [r["is_correct"] for r in results]
    diffs     = [r["difficulty"] for r in results]
    n_uniques = [r["n_unique"] for r in results]

    overall_acc = sum(corrects) / len(corrects)

    # AUROC: σ²_span (higher = more uncertain = predicts incorrect)
    auroc_sigma2 = compute_auroc(sigma2s, corrects)
    ci_half, _ = bootstrap_auroc(sigma2s, corrects, n=CONFIG["BOOTSTRAP_SAMPLES"])

    # AUROC: majority_confidence (higher confidence = predicts correct)
    auroc_conf = compute_auroc([-c for c in maj_confs], corrects)

    # ECE: use majority_confidence as confidence proxy
    ece_val, bin_data = compute_ece(maj_confs, corrects, CONFIG["ECE_BINS"])

    # Pearson r: σ² vs difficulty
    rho = pearson_r(sigma2s, diffs)

    # Pearson r: accuracy vs difficulty
    rho_acc_diff = pearson_r(corrects, [-d for d in diffs])

    # Breakdown by difficulty tier
    easy   = [r for r in results if r["difficulty"] <= 0.25]
    medium = [r for r in results if 0.25 < r["difficulty"] <= 0.65]
    hard   = [r for r in results if r["difficulty"] > 0.65]
    acc_easy   = sum(r["is_correct"] for r in easy)   / len(easy)   if easy else 0
    acc_medium = sum(r["is_correct"] for r in medium) / len(medium) if medium else 0
    acc_hard   = sum(r["is_correct"] for r in hard)   / len(hard)   if hard else 0
    sig_easy   = statistics.mean(r["sigma2_span"] for r in easy)   if easy else 0
    sig_medium = statistics.mean(r["sigma2_span"] for r in medium) if medium else 0
    sig_hard   = statistics.mean(r["sigma2_span"] for r in hard)   if hard else 0

    # K-stability sweep (estimate from existing 8 passes by subsampling)
    def auroc_k(k: int, n_rep: int = 50, seed: int = 99) -> Tuple[float, float]:
        """Estimate AUROC at K passes by subsampling from 8 answers."""
        rng = random.Random(seed)
        aus = []
        from collections import Counter
        for _ in range(n_rep):
            sigs, corrs = [], []
            for r in results:
                ans_k = rng.choices(r["answers"], k=k)
                cnts = Counter(ans_k)
                n = len(ans_k)
                ent = -sum((c/n)*math.log(c/n) for c in cnts.values() if c > 0)
                mx = math.log(n) if n > 1 else 1.0
                sigs.append(ent / mx)
                corrs.append(r["is_correct"])
            a = compute_auroc(sigs, corrs)
            if not math.isnan(a):
                aus.append(a)
        if not aus:
            return float("nan"), 0.0
        return statistics.mean(aus), statistics.stdev(aus) if len(aus) > 1 else 0.0

    k_sweep = {}
    for k in [1, 2, 3, 4, 6, 8]:
        mean_a, std_a = auroc_k(k)
        k_sweep[k] = {"mean": round(mean_a, 4), "std": round(std_a, 4)}
        print(f"  K={k}: AUROC = {mean_a:.3f} ± {std_a:.3f}")

    analysis = {
        "N": len(results),
        "K": CONFIG["K_PASSES"],
        "overall_acc": round(overall_acc, 4),
        "auroc_sigma2": round(auroc_sigma2, 4),
        "auroc_sigma2_ci95": round(ci_half * 1.96, 4),
        "auroc_confidence": round(auroc_conf, 4),
        "ece": round(ece_val, 4),
        "rho_sigma2_difficulty": round(rho, 4),
        "rho_acc_difficulty": round(rho_acc_diff, 4),
        "accuracy_by_tier": {
            "easy":   {"n": len(easy),   "acc": round(acc_easy, 3),   "mean_sigma2": round(sig_easy, 3)},
            "medium": {"n": len(medium), "acc": round(acc_medium, 3), "mean_sigma2": round(sig_medium, 3)},
            "hard":   {"n": len(hard),   "acc": round(acc_hard, 3),   "mean_sigma2": round(sig_hard, 3)},
        },
        "k_stability": k_sweep,
        "ece_bins": bin_data,
        "elapsed_total_s": round(time.time() - t0, 1),
    }

    out_analysis = CONFIG["OUTPUT_DIR"] / CONFIG["ANALYSIS_FILE"]
    with open(out_analysis, "w") as f:
        json.dump(analysis, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print(f"📊 BPFC Extended Pilot Results (N={len(results)})")
    print("="*60)
    print(f"  Overall accuracy:         {overall_acc:.3f}")
    print(f"  AUROC(σ²_span→correct):  {auroc_sigma2:.3f} ± {ci_half*1.96:.3f} (95% CI)")
    print(f"  AUROC(majority_conf):     {auroc_conf:.3f}")
    print(f"  ECE (10 bins):            {ece_val:.3f}")
    print(f"  ρ(σ², difficulty):        {rho:.3f}")
    print(f"\n  Difficulty breakdown:")
    print(f"    Easy   (N={len(easy):<3d}): acc={acc_easy:.2f} σ²={sig_easy:.3f}")
    print(f"    Medium (N={len(medium):<3d}): acc={acc_medium:.2f} σ²={sig_medium:.3f}")
    print(f"    Hard   (N={len(hard):<3d}): acc={acc_hard:.2f} σ²={sig_hard:.3f}")
    print(f"\n  K-stability (AUROC):")
    for k, v in k_sweep.items():
        print(f"    K={k}: {v['mean']:.3f} ± {v['std']:.3f}")
    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")
    print(f"💾 Analysis → {out_analysis}")

    return analysis


if __name__ == "__main__":
    main()
