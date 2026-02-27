"""
BPFC Ensemble Experiment: ALBERT-large-v2 + DistilBERT-base-uncased

Hypothesis: Combining σ²_answer from two architecturally diverse models
(parameter-sharing ALBERT-large vs. distilled BERT) via score-level averaging
will produce a better-calibrated ensemble AUROC > 0.95.

Prior results:
  ALBERT-large-v2 (18M)  : AUROC=0.946 [0.881, 0.994], Cohen's d=2.205
  DistilBERT-base (66M)  : AUROC=0.835 [0.704, 0.939], Cohen's d=1.221

Ensemble method:
  σ²_ens = (σ²_albert + σ²_distilbert) / 2   (equal-weight average)

Also tests:
  - Rank-ensemble: avg of rank-normalized scores
  - Max-ensemble: max(σ²_albert, σ²_distilbert)

Session: Dr. Claw 2026-02-27 05:21 UTC
"""
import time
import json
import random
import math
import sys
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ─── Question Bank (same stratified set, consistent with prior experiments) ───
QUESTION_BANK = [
    # Easy (20)
    ("The capital of France is [MASK].", "paris"),
    ("Water freezes at [MASK] degrees Celsius.", "0"),
    ("The chemical symbol for gold is [MASK].", "au"),
    ("Shakespeare wrote [MASK].", "hamlet"),
    ("The largest planet in our solar system is [MASK].", "jupiter"),
    ("The speed of light is approximately [MASK] km/s.", "300000"),
    ("DNA stands for deoxyribonucleic [MASK].", "acid"),
    ("The currency of Japan is the [MASK].", "yen"),
    ("Albert Einstein developed the theory of [MASK].", "relativity"),
    ("The Earth orbits the [MASK].", "sun"),
    ("The Eiffel Tower is located in [MASK].", "paris"),
    ("Oxygen has atomic number [MASK].", "8"),
    ("The Amazon River is in [MASK].", "brazil"),
    ("The human body has [MASK] bones.", "206"),
    ("Mount Everest is the tallest [MASK] on Earth.", "mountain"),
    ("The Mona Lisa was painted by [MASK].", "leonardo"),
    ("The first US President was George [MASK].", "washington"),
    ("Photosynthesis converts light into [MASK].", "energy"),
    ("The boiling point of water is [MASK] degrees Celsius.", "100"),
    ("The Great Wall is located in [MASK].", "china"),
    # Medium (15)
    ("The Treaty of Versailles was signed in [MASK].", "1919"),
    ("The mitochondria is the [MASK] of the cell.", "powerhouse"),
    ("Isaac Newton formulated the law of [MASK].", "gravity"),
    ("The speed of sound in air is approximately [MASK] m/s.", "343"),
    ("The Pythagorean theorem states a² + b² = [MASK].", "c²"),
    ("Penicillin was discovered by Alexander [MASK].", "fleming"),
    ("The human genome contains approximately [MASK] base pairs.", "3"),
    ("The French Revolution began in [MASK].", "1789"),
    ("The periodic table was created by [MASK].", "mendeleev"),
    ("The speed of a photon in vacuum is [MASK].", "c"),
    ("Copper has atomic symbol [MASK].", "cu"),
    ("The Sistine Chapel ceiling was painted by [MASK].", "michelangelo"),
    ("Alan Turing proposed the [MASK] test.", "turing"),
    ("The hippocampus is associated with [MASK].", "memory"),
    ("The half-life of carbon-14 is approximately [MASK] years.", "5730"),
    # Hard (15)
    ("The Chandrasekhar limit is approximately 1.4 solar [MASK].", "masses"),
    ("Gödel's incompleteness theorem applies to [MASK] systems.", "formal"),
    ("The Krebs cycle produces [MASK] ATP per glucose.", "2"),
    ("The P vs NP problem is classified as [MASK].", "unsolved"),
    ("Avogadro's number is approximately 6.022 × 10 to the [MASK].", "23"),
    ("The Treaty of Westphalia ended the [MASK] Years War.", "thirty"),
    ("The quark model was proposed by [MASK].", "gell-mann"),
    ("Fermat's Last Theorem was proven by [MASK].", "wiles"),
    ("The Coriolis effect influences [MASK] patterns.", "weather"),
    ("CRISPR-Cas9 is a gene [MASK] tool.", "editing"),
    ("The Higgs boson was confirmed in [MASK].", "2012"),
    ("Dark matter constitutes about [MASK] percent of the universe.", "27"),
    ("The Navier-Stokes equations describe fluid [MASK].", "dynamics"),
    ("The entropy of a black hole is proportional to its event [MASK] area.", "horizon"),
    ("The Banach-Tarski paradox involves [MASK] decomposition.", "set"),
]

def get_tier(idx):
    if idx < 20: return "easy"
    elif idx < 35: return "medium"
    else: return "hard"


def compute_sigma2_for_model(model_name, tokenizer, model, questions, K=8, temperature=1.0):
    """
    For each question, run K temperature-sampled MLM passes and compute:
      σ²_answer = NTR (normalized type-token ratio) = len(unique sampled tokens) / K

    This is the same metric used in albert_scale_sweep.py and distilbert_crossval.py
    that produced AUROC=0.946 and 0.835 respectively.

    Interpretation: when the model is uncertain, it samples many different tokens
    (high NTR); when confident, it keeps returning the same token (low NTR).
    High NTR → uncertain → predict wrong.
    """
    results = []
    model.eval()
    mask_token = tokenizer.mask_token  # '<mask>' for ALBERT, '[MASK]' for BERT/DistilBERT

    for q_idx, (question, gold) in enumerate(questions):
        # Replace [MASK] with model's actual mask token
        prompt = question.replace("[MASK]", mask_token)
        inputs = tokenizer(prompt, return_tensors="pt")
        mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        if len(mask_positions) == 0:
            continue

        mask_pos = mask_positions[0].item()
        gold_tokens = tokenizer.encode(gold, add_special_tokens=False)
        if not gold_tokens:
            continue
        gold_token_id = gold_tokens[0]

        sampled_tokens = []
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [1, seq_len, vocab]
            mask_logits = logits[0, mask_pos, :]  # [vocab]
            scaled_logits = mask_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)

            for k in range(K):
                # Temperature-sampled: sample a token at the mask position
                tok_id = torch.multinomial(probs, num_samples=1).item()
                sampled_tokens.append(tok_id)

        # σ²_answer = normalized type-token ratio (NTR)
        sigma2 = len(set(sampled_tokens)) / K  # in [1/K, 1.0]

        # Predicted answer = majority vote of K samples
        majority_id = Counter(sampled_tokens).most_common(1)[0][0]
        predicted = tokenizer.decode([majority_id]).strip().lower()
        gold_lower = gold.lower().strip()
        correct = (gold_lower in predicted or predicted in gold_lower or
                   predicted == gold_lower)

        tier = get_tier(q_idx)
        results.append({
            "question": question,
            "gold": gold,
            "predicted": predicted,
            "sigma2": sigma2,
            "correct": bool(correct),
            "tier": tier,
            "n_unique_tokens": len(set(sampled_tokens)),
            "sampled_token_ids": sampled_tokens,
        })

    return results


def auroc(sigma2_list, correct_list):
    """
    AUROC: Higher σ² = more uncertain = predict wrong.
    We measure: AUROC(σ², is_wrong).
    """
    n = len(sigma2_list)
    pairs = list(zip(sigma2_list, correct_list))
    # Sort by σ² descending (higher σ² = predict wrong)
    pairs.sort(key=lambda x: x[0], reverse=True)
    n_pos = sum(1 for _, c in pairs if not c)  # wrong = positive class
    n_neg = sum(1 for _, c in pairs if c)       # correct = negative class

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp, fp = 0, 0
    auc = 0.0
    prev_fp = 0
    for sigma2, correct in pairs:
        if not correct:  # wrong answer: positive
            tp += 1
        else:             # correct answer: negative
            fp += 1
            # Trapezoid rule: integrate as FP increases by 1
            auc += tp / n_pos * (1 / n_neg)  # simplified

    # Wilcoxon-Mann-Whitney estimator (correct formula)
    correct_sigmas = [s for s, c in zip(sigma2_list, correct_list) if c]
    wrong_sigmas = [s for s, c in zip(sigma2_list, correct_list) if not c]
    if not correct_sigmas or not wrong_sigmas:
        return 0.5
    count = sum(1 for sw in wrong_sigmas for sc in correct_sigmas if sw > sc)
    count += 0.5 * sum(1 for sw in wrong_sigmas for sc in correct_sigmas if sw == sc)
    auc = count / (len(correct_sigmas) * len(wrong_sigmas))
    return auc


def bootstrap_ci(sigma2_list, correct_list, n_boot=1000, seed=42):
    random.seed(seed)
    n = len(sigma2_list)
    boot_aurocs = []
    for _ in range(n_boot):
        idxs = [random.randint(0, n-1) for _ in range(n)]
        s2 = [sigma2_list[i] for i in idxs]
        c = [correct_list[i] for i in idxs]
        a = auroc(s2, c)
        boot_aurocs.append(a)
    boot_aurocs.sort()
    lo = boot_aurocs[int(0.025 * n_boot)]
    hi = boot_aurocs[int(0.975 * n_boot)]
    return lo, hi


def cohens_d(sigma2_list, correct_list):
    correct_s2 = [s for s, c in zip(sigma2_list, correct_list) if c]
    wrong_s2 = [s for s, c in zip(sigma2_list, correct_list) if not c]
    if len(correct_s2) < 2 or len(wrong_s2) < 2:
        return 0.0
    mc = sum(correct_s2) / len(correct_s2)
    mw = sum(wrong_s2) / len(wrong_s2)
    vc = sum((x - mc)**2 for x in correct_s2) / (len(correct_s2) - 1)
    vw = sum((x - mw)**2 for x in wrong_s2) / (len(wrong_s2) - 1)
    sp = math.sqrt((vc + vw) / 2)
    return (mw - mc) / sp if sp > 0 else 0.0


def rank_normalize(scores):
    """Rank-normalize scores to [0,1]: 0 = smallest, 1 = largest."""
    n = len(scores)
    if n <= 1:
        return scores
    indexed = sorted(enumerate(scores), key=lambda x: x[1])
    ranks = [0.0] * n
    for rank, (idx, _) in enumerate(indexed):
        ranks[idx] = rank / (n - 1)
    return ranks


def run_ensemble():
    print("=" * 60)
    print("BPFC Ensemble Experiment: ALBERT-large + DistilBERT")
    print("=" * 60)

    MODELS = [
        ("albert-large-v2", "ALBERT-large-v2 (18M)"),
        ("distilbert-base-uncased", "DistilBERT-base (66M)"),
    ]
    K = 8
    N = len(QUESTION_BANK)  # All 50

    all_results = {}

    for model_name, label in MODELS:
        print(f"\n[{label}] Loading model...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.eval()
        print(f"  Loaded in {time.time()-t0:.1f}s")

        print(f"  Running K={K} passes on N={N} questions...")
        t1 = time.time()
        results = compute_sigma2_for_model(
            model_name, tokenizer, model, QUESTION_BANK, K=K
        )
        elapsed = time.time() - t1
        print(f"  Done in {elapsed:.1f}s")

        sigma2_vals = [r["sigma2"] for r in results]
        correct_vals = [r["correct"] for r in results]
        auc = auroc(sigma2_vals, correct_vals)
        lo, hi = bootstrap_ci(sigma2_vals, correct_vals)
        d = cohens_d(sigma2_vals, correct_vals)
        acc = sum(correct_vals) / len(correct_vals)

        print(f"  AUROC={auc:.3f} [{lo:.3f}, {hi:.3f}], Cohen's d={d:.3f}, Acc={acc:.2f}")

        all_results[model_name] = {
            "label": label,
            "results": results,
            "sigma2": sigma2_vals,
            "correct": correct_vals,
            "auroc": auc,
            "ci_lo": lo,
            "ci_hi": hi,
            "cohens_d": d,
            "accuracy": acc,
        }

    # ─── Build ensembles ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Ensemble Results")
    print("=" * 60)

    albert_s2 = all_results["albert-large-v2"]["sigma2"]
    distil_s2 = all_results["distilbert-base-uncased"]["sigma2"]
    correct_vals = all_results["albert-large-v2"]["correct"]  # same questions, same order

    # Verify alignment
    assert len(albert_s2) == len(distil_s2), "Mismatched lengths!"

    # --- Method 1: Equal-weight average ---
    avg_s2 = [(a + b) / 2 for a, b in zip(albert_s2, distil_s2)]
    avg_auc = auroc(avg_s2, correct_vals)
    avg_lo, avg_hi = bootstrap_ci(avg_s2, correct_vals)
    avg_d = cohens_d(avg_s2, correct_vals)
    print(f"\n[Ensemble-AVG]  AUROC={avg_auc:.3f} [{avg_lo:.3f}, {avg_hi:.3f}], Cohen's d={avg_d:.3f}")

    # --- Method 2: Rank-normalized average ---
    albert_rank = rank_normalize(albert_s2)
    distil_rank = rank_normalize(distil_s2)
    rank_s2 = [(a + b) / 2 for a, b in zip(albert_rank, distil_rank)]
    rank_auc = auroc(rank_s2, correct_vals)
    rank_lo, rank_hi = bootstrap_ci(rank_s2, correct_vals)
    rank_d = cohens_d(rank_s2, correct_vals)
    print(f"[Ensemble-RANK] AUROC={rank_auc:.3f} [{rank_lo:.3f}, {rank_hi:.3f}], Cohen's d={rank_d:.3f}")

    # --- Method 3: Max ensemble ---
    max_s2 = [max(a, b) for a, b in zip(albert_s2, distil_s2)]
    max_auc = auroc(max_s2, correct_vals)
    max_lo, max_hi = bootstrap_ci(max_s2, correct_vals)
    max_d = cohens_d(max_s2, correct_vals)
    print(f"[Ensemble-MAX]  AUROC={max_auc:.3f} [{max_lo:.3f}, {max_hi:.3f}], Cohen's d={max_d:.3f}")

    # ─── Per-tier breakdown for best ensemble ────────────────────────────────
    tiers = [r["tier"] for r in all_results["albert-large-v2"]["results"]]  # same for both models
    best_s2 = rank_s2 if rank_auc >= max(avg_auc, max_auc) else (avg_s2 if avg_auc >= max_auc else max_s2)
    best_label = "RANK" if rank_auc >= max(avg_auc, max_auc) else ("AVG" if avg_auc >= max_auc else "MAX")

    print(f"\nPer-tier breakdown (best ensemble: {best_label}):")
    for tier in ["easy", "medium", "hard"]:
        tier_s2 = [s for s, t in zip(best_s2, tiers) if t == tier]
        tier_c = [c for c, t in zip(correct_vals, tiers) if t == tier]
        tier_auc = auroc(tier_s2, tier_c)
        tier_acc = sum(tier_c) / len(tier_c)
        print(f"  {tier:8s}: AUROC={tier_auc:.3f}, Acc={tier_acc:.2f}, N={len(tier_s2)}")

    # ─── Save results ─────────────────────────────────────────────────────────
    output = {
        "experiment": "BPFC Ensemble: ALBERT-large + DistilBERT",
        "date": "2026-02-27",
        "session": "Dr. Claw #20",
        "K": K,
        "N": N,
        "individual": {
            name: {
                "label": all_results[name]["label"],
                "auroc": all_results[name]["auroc"],
                "ci_lo": all_results[name]["ci_lo"],
                "ci_hi": all_results[name]["ci_hi"],
                "cohens_d": all_results[name]["cohens_d"],
                "accuracy": all_results[name]["accuracy"],
            }
            for name in all_results
        },
        "ensemble_avg": {
            "auroc": avg_auc,
            "ci_lo": avg_lo,
            "ci_hi": avg_hi,
            "cohens_d": avg_d,
        },
        "ensemble_rank": {
            "auroc": rank_auc,
            "ci_lo": rank_lo,
            "ci_hi": rank_hi,
            "cohens_d": rank_d,
        },
        "ensemble_max": {
            "auroc": max_auc,
            "ci_lo": max_lo,
            "ci_hi": max_hi,
            "cohens_d": max_d,
        },
        "best_ensemble": best_label,
        "per_tier_best": {
            tier: {
                "auroc": auroc(
                    [s for s, t in zip(best_s2, tiers) if t == tier],
                    [c for c, t in zip(correct_vals, tiers) if t == tier]
                ),
                "accuracy": sum(c for c, t in zip(correct_vals, tiers) if t == tier) / len([c for c, t in zip(correct_vals, tiers) if t == tier]),
            }
            for tier in ["easy", "medium", "hard"]
        }
    }

    out_path = "results/ensemble_bpfc_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    t_start = time.time()
    results = run_ensemble()
    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"ALBERT-large-v2:   AUROC={results['individual']['albert-large-v2']['auroc']:.3f}")
    print(f"DistilBERT-base:   AUROC={results['individual']['distilbert-base-uncased']['auroc']:.3f}")
    print(f"Ensemble-AVG:      AUROC={results['ensemble_avg']['auroc']:.3f} [{results['ensemble_avg']['ci_lo']:.3f}, {results['ensemble_avg']['ci_hi']:.3f}]")
    print(f"Ensemble-RANK:     AUROC={results['ensemble_rank']['auroc']:.3f} [{results['ensemble_rank']['ci_lo']:.3f}, {results['ensemble_rank']['ci_hi']:.3f}]")
    print(f"Ensemble-MAX:      AUROC={results['ensemble_max']['auroc']:.3f} [{results['ensemble_max']['ci_lo']:.3f}, {results['ensemble_max']['ci_hi']:.3f}]")
    print(f"Best ensemble:     {results['best_ensemble']}")
