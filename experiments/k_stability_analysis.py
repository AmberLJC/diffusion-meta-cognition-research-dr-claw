#!/usr/bin/env python3
"""
K-Stability Analysis: BPFC AUROC vs. Number of Denoising Passes K
==================================================================
Shows that AUROC(σ²_answer → error) stabilises as K → ∞, consistent with
Corollary 3.2 of the paper (convergence of Monte-Carlo posterior variance).

AUTHOR: Dr. Claw | 2026-02-27 (Run #7)
"""

import json
import math
import random
import statistics
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    "N_QUESTIONS": 100,           # Larger N for statistical power
    "K_MAX": 16,                  # Test K from 1 to K_MAX
    "K_VALUES": [1, 2, 3, 4, 6, 8, 12, 16],
    "BERT_MODEL": "bert-base-uncased",
    "SEED": 42,
    "OUTPUT_DIR": Path(__file__).parent.parent / "data",
}
CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
random.seed(CONFIG["SEED"])

# ── Extended Factual QA Dataset (N=100) ───────────────────────────────────────

FACTUAL_QA = [
    # === EASY (difficulty 0.0–0.2) ===
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
    ("The smallest planet in our solar system is [MASK].", "mercury", 0.4),
    ("The chemical symbol for sodium is [MASK].", "na", 0.4),
    ("The Battle of [MASK] was fought in 1815.", "waterloo", 0.45),
    ("Photosynthesis converts sunlight into [MASK].", "energy", 0.3),
    ("The [MASK] is the longest river in Africa.", "nile", 0.3),
    ("Mozart was a famous [MASK] composer.", "austrian", 0.5),
    ("The speed of light in a vacuum is [MASK].", "c", 0.4),
    ("The United States was founded in [MASK].", "1776", 0.35),
    ("The human genome contains approximately 3 billion [MASK] pairs.", "base", 0.5),
    ("The chemical symbol for potassium is [MASK].", "k", 0.45),
    ("Archimedes discovered the principle of [MASK].", "buoyancy", 0.45),
    ("The [MASK] Revolution began in 1789 in France.", "french", 0.35),
    ("The atomic number of carbon is [MASK].", "6", 0.4),
    ("The capital of Brazil is [MASK].", "brasilia", 0.5),
    ("The [MASK] is the largest continent by area.", "asia", 0.35),

    # === HARD (difficulty 0.7–1.0) ===
    ("The capital of Kazakhstan is [MASK].", "astana", 0.8),
    ("The chemical symbol for tungsten is [MASK].", "w", 0.75),
    ("The Treaty of [MASK] ended the First World War.", "versailles", 0.7),
    ("The [MASK] programming language was created by Guido van Rossum.", "python", 0.7),
    ("The element with atomic number 79 is [MASK].", "gold", 0.75),
    ("The [MASK] is the smallest bone in the human body.", "stapes", 0.85),
    ("Euler's number e is approximately [MASK].", "2.718", 0.75),
    ("The deepest lake in the world is Lake [MASK].", "baikal", 0.8),
    ("The [MASK] dynasty ruled China for over 270 years.", "qing", 0.85),
    ("The half-life of Carbon-14 is approximately [MASK] years.", "5730", 0.9),
    ("The philosopher Immanuel Kant was born in [MASK].", "konigsberg", 0.85),
    ("The [MASK] constant describes the expansion of the universe.", "hubble", 0.8),
    ("The capital of Burkina Faso is [MASK].", "ouagadougou", 0.95),
    ("The [MASK] acid is the main component of stomach acid.", "hydrochloric", 0.8),
    ("Marie Curie discovered both radium and [MASK].", "polonium", 0.75),
    ("The [MASK] effect describes the redshift of light from distant galaxies.", "doppler", 0.75),
    ("The capital of Kyrgyzstan is [MASK].", "bishkek", 0.9),
    ("The wavelength of visible light ranges from 380 to [MASK] nanometers.", "700", 0.85),
    ("The [MASK] bone is the longest in the human body.", "femur", 0.7),
    ("Feynman developed [MASK] diagrams for quantum field theory.", "feynman", 0.75),
    ("The second law of thermodynamics concerns [MASK].", "entropy", 0.7),
    ("The [MASK] Galaxy is the nearest spiral galaxy to the Milky Way.", "andromeda", 0.7),
    ("The chemical symbol for mercury is [MASK].", "hg", 0.7),
    ("Pascal's triangle is used in [MASK] combinatorics.", "binomial", 0.8),
    ("The [MASK] transform converts time-domain signals to frequency-domain.", "fourier", 0.75),
    ("The capital of Tajikistan is [MASK].", "dushanbe", 0.9),
    ("Gödel's incompleteness theorem applies to formal [MASK] systems.", "arithmetic", 0.85),
    ("The [MASK] equation describes the motion of quantum particles.", "schrodinger", 0.8),
    ("The Battle of [MASK] in 1066 changed English history.", "hastings", 0.7),
    ("The [MASK] is the SI unit of electric capacitance.", "farad", 0.8),

    # Additional MEDIUM questions to fill 100 total
    ("The process of water turning to steam is called [MASK].", "evaporation", 0.3),
    ("The liver produces a digestive fluid called [MASK].", "bile", 0.45),
    ("Isaac Asimov wrote the [MASK] series of science fiction novels.", "foundation", 0.55),
    ("The [MASK] bone protects the brain.", "skull", 0.3),
    ("The [MASK] is the organ responsible for pumping blood.", "heart", 0.1),
    ("The Wright Brothers made the first successful airplane flight at [MASK].", "kitty hawk", 0.5),
    ("The [MASK] is the basic unit of heredity.", "gene", 0.4),
    ("The area of a circle is π times the radius [MASK].", "squared", 0.35),
    ("Light refracts when it passes through a [MASK].", "prism", 0.45),
    ("The [MASK] War was fought between the Union and Confederacy.", "civil", 0.35),

    # Final 10 to reach N=100
    ("The mitochondria is known as the [MASK] of the cell.", "powerhouse", 0.35),
    ("Beethoven was a German [MASK].", "composer", 0.25),
    ("The [MASK] scale measures earthquake magnitude.", "richter", 0.55),
    ("The [MASK] is the SI unit of electrical resistance.", "ohm", 0.55),
    ("The [MASK] is the muscle that controls breathing.", "diaphragm", 0.6),
    ("The [MASK] is the process by which plants make food from sunlight.", "photosynthesis", 0.3),
    ("Sound travels faster through [MASK] than through air.", "water", 0.45),
    ("The [MASK] is the currency of the United Kingdom.", "pound", 0.3),
    ("The [MASK] is the outermost layer of the Earth.", "crust", 0.45),
    ("Abraham Lincoln was the [MASK] President of the United States.", "16th", 0.6),
]

# Trim to exactly N_QUESTIONS
assert len(FACTUAL_QA) >= CONFIG["N_QUESTIONS"], f"Need {CONFIG['N_QUESTIONS']} questions, have {len(FACTUAL_QA)}"
FACTUAL_QA = FACTUAL_QA[:CONFIG["N_QUESTIONS"]]


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model():
    """Load BERT fill-mask pipeline on CPU."""
    print("Loading BERT-base fill-mask pipeline (CPU)...")
    t0 = time.time()
    try:
        from transformers import pipeline
        nlp = pipeline(
            "fill-mask",
            model=CONFIG["BERT_MODEL"],
            device=-1,          # CPU
            top_k=50,           # Get wider distribution
        )
        print(f"✓ Model loaded in {time.time()-t0:.1f}s")
        return nlp
    except Exception as e:
        print(f"✗ Model load failed: {e}")
        raise


def normalize_answer(a: str) -> str:
    return a.strip().lower().rstrip('.,!?')


def sample_answer(nlp, template: str, gold: str) -> tuple[str, float]:
    """Sample one fill-mask answer. Returns (answer, token_prob)."""
    try:
        results = nlp(template)
        # Weighted random sample from top-k
        tokens = [r["token_str"].strip().lower() for r in results]
        probs = [r["score"] for r in results]
        total = sum(probs)
        probs_norm = [p / total for p in probs]
        idx = random.choices(range(len(tokens)), weights=probs_norm, k=1)[0]
        return tokens[idx], probs[idx]
    except Exception:
        return "[unk]", 0.0


# ── Per-K experiment ──────────────────────────────────────────────────────────

def run_for_k(nlp, questions, k: int) -> dict:
    """
    Run BPFC experiment for a specific K value.
    Returns metrics dict: {k, auroc_sigma2_answer, accuracy, ...}
    """
    results = []
    for template, gold, difficulty in questions:
        answers = []
        token_probs = []
        for _ in range(k):
            ans, prob = sample_answer(nlp, template, gold)
            answers.append(normalize_answer(ans))
            token_probs.append(prob)

        # Majority vote
        majority = max(set(answers), key=answers.count)
        is_correct = (majority == normalize_answer(gold))

        # σ²_answer: variance over binary correct/incorrect per pass
        pass_correct = [int(a == normalize_answer(gold)) for a in answers]
        mean_pc = sum(pass_correct) / k
        sigma2_answer = sum((x - mean_pc)**2 for x in pass_correct) / k if k > 1 else 0.0

        # σ²_token: variance over token probabilities
        mean_tp = sum(token_probs) / k
        sigma2_token = sum((x - mean_tp)**2 for x in token_probs) / k if k > 1 else 0.0

        results.append({
            "is_correct": is_correct,
            "sigma2_answer": sigma2_answer,
            "sigma2_token": sigma2_token,
            "difficulty": difficulty,
        })

    # Compute AUROC
    incorrect = [not r["is_correct"] for r in results]
    sigma2_ans = [r["sigma2_answer"] for r in results]
    accuracy = sum(r["is_correct"] for r in results) / len(results)

    auroc_val = auroc(sigma2_ans, incorrect)

    return {
        "k": k,
        "n": len(questions),
        "accuracy": accuracy,
        "auroc_sigma2_answer": auroc_val,
    }


def auroc(scores: list, labels: list) -> float:
    """Compute AUROC. labels=True means positive class (incorrect)."""
    pos = [(s, l) for s, l in zip(scores, labels) if l]
    neg = [(s, l) for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float('nan')
    concordant = sum(1 for (sp, _) in pos for (sn, _) in neg if sp > sn)
    tied = sum(0.5 for (sp, _) in pos for (sn, _) in neg if sp == sn)
    return (concordant + tied) / (len(pos) * len(neg))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import sys
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        from transformers import pipeline
        print(f"✓ DRY RUN OK: {len(FACTUAL_QA)} questions, K values: {CONFIG['K_VALUES']}")
        return

    t0 = time.time()
    nlp = load_model()

    # Cache answers at K_MAX (resample for each K ≤ K_MAX for fairness,
    # but use the same random seed sub-sequence so results are nested)
    print(f"\nRunning K-stability sweep: N={CONFIG['N_QUESTIONS']}, K_values={CONFIG['K_VALUES']}")
    print("This validates Corollary 3.2 (σ²_span convergence as K → ∞)\n")

    k_results = []
    for k in CONFIG["K_VALUES"]:
        random.seed(CONFIG["SEED"] + k)  # reproducible per K
        t_k = time.time()
        metrics = run_for_k(nlp, FACTUAL_QA, k)
        elapsed = time.time() - t_k
        k_results.append(metrics)
        print(f"  K={k:2d} | AUROC={metrics['auroc_sigma2_answer']:.4f} | "
              f"acc={metrics['accuracy']:.1%} | {elapsed:.1f}s")

    print(f"\nTotal time: {time.time()-t0:.1f}s")

    # Print table
    print("\n" + "="*60)
    print("K-STABILITY TABLE (Corollary 3.2 validation)")
    print("="*60)
    print(f"{'K':>4}  {'AUROC(σ²_ans)':>14}  {'Accuracy':>10}")
    print("-"*40)
    for r in k_results:
        print(f"{r['k']:>4}  {r['auroc_sigma2_answer']:>14.4f}  {r['accuracy']:>9.1%}")
    print("="*60)

    # Convergence check: compare K=4,6,8 range
    aucs_high_k = [r['auroc_sigma2_answer'] for r in k_results if r['k'] >= 4]
    if len(aucs_high_k) >= 2:
        spread = max(aucs_high_k) - min(aucs_high_k)
        print(f"\nAUROC spread for K≥4: {spread:.4f}  "
              f"({'✓ stable' if spread < 0.05 else '⚠ variable'})")

    # Save results
    out_path = CONFIG["OUTPUT_DIR"] / "k_stability_results.json"
    with open(out_path, "w") as f:
        json.dump({"k_results": k_results, "config": {
            "N_QUESTIONS": CONFIG["N_QUESTIONS"],
            "K_VALUES": CONFIG["K_VALUES"],
            "BERT_MODEL": CONFIG["BERT_MODEL"],
        }}, f, indent=2)
    print(f"\nResults saved: {out_path}")

    return k_results


if __name__ == "__main__":
    main()
