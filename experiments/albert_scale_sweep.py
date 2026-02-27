"""
BPFC Scale Sweep: ALBERT-base-v2 and ALBERT-large-v2
Extends the 3-way architecture comparison (DistilBERT/BERT/RoBERTa) to 5 models.

Hypothesis: Inverse relationship between parameter count and AUROC(σ²_answer)
ALBERT's parameter-sharing means same Transformer depth but far fewer stored params.
This tests whether *stored parameters* or *forward-pass capacity* drives signal quality.

Session: Dr. Claw 2026-02-27 05:06 UTC
"""
import time
import json
import random
import math
import sys

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ─── Question Bank (same stratified set used in prior experiments) ───────────
QUESTION_BANK = [
    # Easy (widely known facts)
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
    # Medium (requires some knowledge)
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
    # Hard (specialist knowledge)
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

# Tier labels for the 50-question bank (20 easy, 15 medium, 15 hard)
def get_tier(idx):
    if idx < 20: return "easy"
    elif idx < 35: return "medium"
    else: return "hard"

# ─── AUROC (Mann-Whitney U) ──────────────────────────────────────────────────
def auroc(scores_pos, scores_neg):
    """AUROC via Mann-Whitney U. Positive = correct answers (want low σ²)."""
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    u = sum(1.0 if p < n else (0.5 if p == n else 0.0)
            for p in scores_pos for n in scores_neg)
    return u / (n_pos * n_neg)

def bootstrap_auroc_ci(scores_pos, scores_neg, n_boot=500, seed=42):
    rng = random.Random(seed)
    vals = []
    combined_pos = list(scores_pos)
    combined_neg = list(scores_neg)
    for _ in range(n_boot):
        s_pos = [rng.choice(combined_pos) for _ in combined_pos]
        s_neg = [rng.choice(combined_neg) for _ in combined_neg]
        vals.append(auroc(s_pos, s_neg))
    vals.sort()
    lo = vals[int(0.025 * n_boot)]
    hi = vals[int(0.975 * n_boot)]
    return lo, hi

def cohens_d(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    va = sum((x-ma)**2 for x in a)/(len(a)-1) if len(a)>1 else 0
    vb = sum((x-mb)**2 for x in b)/(len(b)-1) if len(b)>1 else 0
    sp = math.sqrt((va + vb) / 2) if (va+vb) > 0 else 1.0
    return (mb - ma) / sp  # positive = wrong answers have higher σ²


# ─── BPFC Pilot for a single model ──────────────────────────────────────────
def run_bpfc_model(model_name: str, K: int = 8, temperature: float = 1.0, 
                   n_questions: int = 50, seed: int = 42):
    """Run BPFC (K-pass MLM variance) for a given masked LM."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"K={K}, T={temperature}, N={n_questions}")
    print('='*60)

    t0 = time.time()
    random.seed(seed)

    # Load model & tokenizer
    print(f"Loading {model_name}...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    print(f"done ({time.time()-t0:.1f}s)")

    # ALBERT uses <mask> not [MASK] — handle both
    mask_token = tokenizer.mask_token
    print(f"  mask_token = '{mask_token}'")

    results = []
    bank = QUESTION_BANK[:n_questions]

    for idx, (template, gold) in enumerate(bank):
        tier = get_tier(idx)
        # Replace [MASK] with model's actual mask token
        prompt = template.replace("[MASK]", mask_token)

        inputs = tokenizer(prompt, return_tensors="pt")
        mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        if len(mask_positions) == 0:
            # Fallback: find mask position manually
            tokens = tokenizer.tokenize(prompt)
            mask_positions = torch.tensor([i+1 for i, t in enumerate(tokens) if mask_token in t])
            if len(mask_positions) == 0:
                print(f"  Warning: no mask token at idx={idx}, skipping")
                continue

        mask_pos = mask_positions[0].item()

        # K-pass sampling: collect top-1 token for each pass
        sampled_tokens = []
        sigma2_values = []

        with torch.no_grad():
            logits = model(**inputs).logits  # [1, seq_len, vocab]
            answer_logits = logits[0, mask_pos, :]  # [vocab]

            for k in range(K):
                # Temperature sampling
                scaled = answer_logits / temperature
                probs = torch.softmax(scaled, dim=-1)

                # Sample token
                tok_id = torch.multinomial(probs, 1).item()
                sampled_tokens.append(tok_id)

                # σ²(pass k) = entropy proxy: -log p(sampled token)
                sigma2_values.append(-math.log(probs[tok_id].item() + 1e-10))

        # σ²_answer: variance over sampled token IDs (type diversity)
        token_id_variance = len(set(sampled_tokens)) / K  # Normalized type-token ratio
        sigma2_answer = token_id_variance

        # Mode-based prediction: majority vote
        from collections import Counter
        mode_id = Counter(sampled_tokens).most_common(1)[0][0]
        predicted = tokenizer.decode([mode_id]).strip().lower()
        gold_lower = gold.lower().strip()

        # Exact or partial match
        correct = (gold_lower in predicted or predicted in gold_lower or
                   predicted == gold_lower)

        results.append({
            "idx": idx,
            "tier": tier,
            "gold": gold,
            "predicted": predicted,
            "correct": correct,
            "sigma2_answer": sigma2_answer,
            "sigma2_mean": sum(sigma2_values) / len(sigma2_values),
            "n_unique_tokens": len(set(sampled_tokens)),
        })

        if (idx + 1) % 10 == 0:
            acc_so_far = sum(r["correct"] for r in results) / len(results)
            print(f"  [{idx+1}/{n_questions}] acc={acc_so_far:.2f}, "
                  f"last σ²={sigma2_answer:.3f}, pred='{predicted}' ({'✓' if correct else '✗'})")

    elapsed = time.time() - t0
    n = len(results)
    accuracy = sum(r["correct"] for r in results) / n

    # AUROC
    pos = [r["sigma2_answer"] for r in results if r["correct"]]
    neg = [r["sigma2_answer"] for r in results if not r["correct"]]
    auc = auroc(pos, neg)
    ci_lo, ci_hi = bootstrap_auroc_ci(pos, neg)
    d = cohens_d(pos, neg)

    # Tier breakdown
    tier_stats = {}
    for tier in ["easy", "medium", "hard"]:
        tier_r = [r for r in results if r["tier"] == tier]
        if tier_r:
            tier_stats[tier] = {
                "n": len(tier_r),
                "acc": sum(r["correct"] for r in tier_r) / len(tier_r),
                "mean_sigma2": sum(r["sigma2_answer"] for r in tier_r) / len(tier_r),
            }

    print(f"\n{'─'*50}")
    print(f"  N={n}, Accuracy={accuracy:.3f}")
    print(f"  AUROC(σ²_answer) = {auc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Cohen's d = {d:.3f}")
    print(f"  Mean σ² correct={sum(pos)/len(pos):.3f}, wrong={sum(neg)/len(neg):.3f}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return {
        "model": model_name,
        "n": n,
        "k": K,
        "temperature": temperature,
        "accuracy": accuracy,
        "elapsed_s": round(elapsed, 1),
        "auroc": auc,
        "auroc_ci_lo": ci_lo,
        "auroc_ci_hi": ci_hi,
        "cohens_d": d,
        "mean_sigma2_correct": sum(pos)/len(pos) if pos else 0,
        "mean_sigma2_wrong": sum(neg)/len(neg) if neg else 0,
        "tier_breakdown": tier_stats,
    }


if __name__ == "__main__":
    results = {}

    # Run both ALBERT variants
    for model_name in ["albert-base-v2", "albert-large-v2"]:
        try:
            r = run_bpfc_model(model_name, K=8, temperature=1.0, n_questions=50)
            results[model_name] = r
        except Exception as e:
            print(f"\nERROR on {model_name}: {e}")
            import traceback; traceback.print_exc()

    # Print consolidated comparison
    print("\n" + "="*70)
    print("5-WAY ARCHITECTURE COMPARISON: Scale vs AUROC(σ²_answer)")
    print("="*70)

    # Prior results
    prior = {
        "distilbert-base-uncased": {"params_M": 66, "auroc": 0.835, "ci_lo": 0.704, "ci_hi": 0.939, "cohens_d": 1.221},
        "bert-base-uncased":       {"params_M": 110, "auroc": 0.791, "ci_lo": 0.639, "ci_hi": 0.927, "cohens_d": 1.626},
        "roberta-large":           {"params_M": 355, "auroc": 0.642, "ci_lo": 0.463, "ci_hi": 0.802, "cohens_d": 0.425},
    }

    # ALBERT parameter counts (effective stored params, not forward-pass ops)
    albert_params = {
        "albert-base-v2": 12,    # 12M effective (parameter-shared transformer)
        "albert-large-v2": 18,   # 18M effective
    }

    all_models = []
    for mname, mdata in prior.items():
        all_models.append({
            "name": mname,
            "params_M": mdata["params_M"],
            "auroc": mdata["auroc"],
            "ci_lo": mdata["ci_lo"],
            "ci_hi": mdata["ci_hi"],
            "cohens_d": mdata["cohens_d"],
            "source": "prior_session",
        })
    for mname, r in results.items():
        all_models.append({
            "name": mname,
            "params_M": albert_params.get(mname, 0),
            "auroc": r["auroc"],
            "ci_lo": r["auroc_ci_lo"],
            "ci_hi": r["auroc_ci_hi"],
            "cohens_d": r["cohens_d"],
            "accuracy": r["accuracy"],
            "source": "this_session",
        })

    # Sort by params ascending
    all_models.sort(key=lambda x: x["params_M"])

    print(f"\n{'Model':<30} {'Params(M)':>10} {'AUROC':>8} {'95% CI':>18} {'d':>6}")
    print("-"*80)
    for m in all_models:
        ci = f"[{m['ci_lo']:.3f}, {m['ci_hi']:.3f}]"
        print(f"{m['name']:<30} {m['params_M']:>10} {m['auroc']:>8.3f} {ci:>18} {m['cohens_d']:>6.3f}")

    # Spearman correlation: params vs AUROC
    params_list = [m["params_M"] for m in all_models]
    auroc_list  = [m["auroc"] for m in all_models]
    n = len(params_list)
    # Rank
    def rank_list(lst):
        sorted_idx = sorted(range(len(lst)), key=lambda i: lst[i])
        ranks = [0]*len(lst)
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = rank + 1
        return ranks
    rp = rank_list(params_list)
    ra = rank_list(auroc_list)
    d2 = sum((rp[i]-ra[i])**2 for i in range(n))
    spearman = 1 - 6*d2 / (n*(n**2-1))
    print(f"\nSpearman ρ(params, AUROC) = {spearman:.3f}  (n={n})")
    if spearman < -0.5:
        print("→ CONFIRMED: Inverse scale relationship (more params → lower AUROC)")
    elif spearman > 0.5:
        print("→ POSITIVE scale relationship (more params → higher AUROC)")
    else:
        print("→ WEAK or NO scale relationship")

    # Save combined results
    output = {
        "session_date": "2026-02-27",
        "spearman_params_vs_auroc": spearman,
        "n_models": len(all_models),
        "all_models": all_models,
        "new_results": results,
    }
    out_path = "/home/azureuser/research/diffusion-meta-cognition-research-dr-claw/results/albert_scale_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print("\n✅ ALBERT scale sweep complete")
