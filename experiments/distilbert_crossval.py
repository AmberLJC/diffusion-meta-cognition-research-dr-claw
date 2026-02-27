#!/usr/bin/env python3
"""
DistilBERT Cross-Architecture Validation of BPFC
=================================================
Third leg of the 3-way architecture comparison:
  DistilBERT-base (66M)  ← NEW
  BERT-base-uncased (110M)  ← from bert_cpu_pilot.py
  RoBERTa-large (355M)  ← from roberta_corrected.py

Method: K=8 temperature-sampled MLM passes → σ²_answer → AUROC vs accuracy

Run time: ~60-90s on CPU (66M params)
"""

import json
import time
import random
import math
from collections import Counter

import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
import torch

random.seed(42)
np.random.seed(42)

# ─── Question Bank (same stratified 50-question bank as bert_cpu_pilot) ───

QUESTIONS = [
    # EASY (well-known facts)
    {"q": "The capital of France is [MASK].",         "answers": ["paris"],       "tier": "easy"},
    {"q": "The chemical symbol for gold is [MASK].",   "answers": ["au"],          "tier": "easy"},
    {"q": "Water boils at 100 degrees [MASK].",        "answers": ["celsius","c"], "tier": "easy"},
    {"q": "The planet closest to the Sun is [MASK].",  "answers": ["mercury"],     "tier": "easy"},
    {"q": "Shakespeare wrote [MASK] about a Danish prince.", "answers": ["hamlet"], "tier": "easy"},
    {"q": "The Great Wall is located in [MASK].",      "answers": ["china"],       "tier": "easy"},
    {"q": "Albert Einstein won the Nobel Prize in [MASK].", "answers": ["physics"], "tier": "easy"},
    {"q": "The human body has [MASK] bones.",          "answers": ["206"],         "tier": "easy"},
    {"q": "Mount Everest is the tallest [MASK] on Earth.", "answers": ["mountain", "peak"], "tier": "easy"},
    {"q": "The Mona Lisa was painted by Leonardo da [MASK].", "answers": ["vinci"], "tier": "easy"},
    {"q": "H2O is the chemical formula for [MASK].",   "answers": ["water"],       "tier": "easy"},
    {"q": "The speed of light is approximately 300,000 kilometers per [MASK].", "answers": ["second"], "tier": "easy"},
    {"q": "Neil Armstrong was the first person to walk on the [MASK].", "answers": ["moon"], "tier": "easy"},
    {"q": "The largest ocean on Earth is the [MASK] Ocean.", "answers": ["pacific"], "tier": "easy"},
    {"q": "Rome is the capital of [MASK].",            "answers": ["italy"],       "tier": "easy"},
    {"q": "The chemical symbol for oxygen is [MASK].", "answers": ["o"],           "tier": "easy"},
    {"q": "The Eiffel Tower is located in [MASK].",    "answers": ["paris","france"], "tier": "easy"},
    {"q": "The sun is a [MASK].",                      "answers": ["star"],        "tier": "easy"},
    {"q": "Python is a programming [MASK].",           "answers": ["language"],    "tier": "easy"},
    {"q": "The Amazon is the world's largest [MASK].", "answers": ["river","rainforest"], "tier": "easy"},

    # MEDIUM (less obvious)
    {"q": "The capital of Australia is [MASK].",       "answers": ["canberra"],    "tier": "medium"},
    {"q": "The element with atomic number 79 is [MASK].", "answers": ["gold"],     "tier": "medium"},
    {"q": "The Treaty of [MASK] ended World War I.",   "answers": ["versailles"],  "tier": "medium"},
    {"q": "The inventor of the telephone was Alexander Graham [MASK].", "answers": ["bell"], "tier": "medium"},
    {"q": "The speed of sound in air is approximately [MASK] meters per second.", "answers": ["343","340"], "tier": "medium"},
    {"q": "The Battle of [MASK] was fought in 1815.",  "answers": ["waterloo"],    "tier": "medium"},
    {"q": "Mitosis is the process of [MASK] division.", "answers": ["cell"],       "tier": "medium"},
    {"q": "The Pythagorean theorem relates the sides of a [MASK] triangle.", "answers": ["right"], "tier": "medium"},
    {"q": "Marie Curie was born in [MASK].",           "answers": ["poland","warsaw"], "tier": "medium"},
    {"q": "The Nobel Prize in Literature 2020 was awarded to Louise [MASK].", "answers": ["gluck", "glück"], "tier": "medium"},
    {"q": "The chemical symbol for sodium is [MASK].", "answers": ["na"],          "tier": "medium"},
    {"q": "The Battle of Thermopylae involved the [MASK] Spartans.", "answers": ["300"], "tier": "medium"},
    {"q": "DNA stands for [MASK] acid.",               "answers": ["deoxyribonucleic","deoxyribo"], "tier": "medium"},
    {"q": "The Suez Canal connects the Red Sea to the [MASK] Sea.", "answers": ["mediterranean"], "tier": "medium"},
    {"q": "Pluto was reclassified as a dwarf [MASK] in 2006.", "answers": ["planet"], "tier": "medium"},

    # HARD (obscure or ambiguous)
    {"q": "The capital of Kazakhstan is [MASK].",      "answers": ["astana","nursultan"], "tier": "hard"},
    {"q": "The author of 'The Name of the Rose' is Umberto [MASK].", "answers": ["eco"], "tier": "hard"},
    {"q": "The element with atomic number 92 is [MASK].", "answers": ["uranium"],  "tier": "hard"},
    {"q": "The Taiping Rebellion occurred in [MASK].", "answers": ["china"],       "tier": "hard"},
    {"q": "The philosopher who wrote 'Being and Nothingness' is Jean-Paul [MASK].", "answers": ["sartre"], "tier": "hard"},
    {"q": "The Treaty of [MASK] established the European Union.",  "answers": ["maastricht"], "tier": "hard"},
    {"q": "The Haversine formula computes distances on a [MASK].", "answers": ["sphere"], "tier": "hard"},
    {"q": "Quarks were first proposed by Murray [MASK].", "answers": ["gell-mann","gellmann"], "tier": "hard"},
    {"q": "The Battle of [MASK] was the first major land battle of the American Civil War.", "answers": ["bull run","manassas"], "tier": "hard"},
    {"q": "The Rashomon effect refers to [MASK] accounts of the same event.", "answers": ["contradictory","conflicting"], "tier": "hard"},
    {"q": "The Fermat point minimizes the total [MASK] from triangle vertices.", "answers": ["distance"], "tier": "hard"},
    {"q": "The longest river in Africa is the [MASK].", "answers": ["nile"],       "tier": "hard"},
    {"q": "The Bohr radius is the most probable distance from the [MASK] in hydrogen.", "answers": ["nucleus","proton"], "tier": "hard"},
    {"q": "The author of 'One Hundred Years of Solitude' is Gabriel Garcia [MASK].", "answers": ["marquez","márquez"], "tier": "hard"},
    {"q": "The first synthetic dye was [MASK], discovered by Perkin in 1856.", "answers": ["mauveine","mauve"], "tier": "hard"},
]

print(f"Question bank: {len(QUESTIONS)} questions")


def load_distilbert():
    """Load DistilBERT with MLM head."""
    print("Loading distilbert-base-uncased (66M)...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    print(f"  Loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return tokenizer, model


def bpfc_single(question: str, tokenizer, model, K: int = 8, temperature: float = 1.0, top_k: int = 50):
    """
    Run K temperature-sampled MLM passes on a question.
    Returns: list of K sampled tokens at [MASK] position.
    """
    inputs = tokenizer(question, return_tensors="pt")
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[1]

    if len(mask_positions) == 0:
        return [], []

    mask_pos = mask_positions[0].item()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, mask_pos, :]  # (vocab_size,)

    # Apply temperature
    scaled_logits = logits / temperature
    # Top-k filter
    top_k_vals, top_k_ids = torch.topk(scaled_logits, top_k)
    probs = torch.softmax(top_k_vals, dim=-1).numpy()
    top_k_ids = top_k_ids.numpy()

    # Sample K times
    sampled = np.random.choice(len(probs), size=K, replace=True, p=probs)
    sampled_tokens = [tokenizer.convert_ids_to_tokens(int(top_k_ids[s])) for s in sampled]
    sampled_ids = [int(top_k_ids[s]) for s in sampled]

    return sampled_tokens, sampled_ids


def compute_sigma2(token_ids: list) -> float:
    """Compute variance of one-hot token distribution = 1 - sum(p_i^2)."""
    if not token_ids:
        return 0.0
    counts = Counter(token_ids)
    n = len(token_ids)
    probs = [c / n for c in counts.values()]
    return 1.0 - sum(p ** 2 for p in probs)


def majority_token(tokens: list) -> str:
    """Return most-common token."""
    if not tokens:
        return ""
    c = Counter(tokens)
    return c.most_common(1)[0][0]


def is_correct(predicted_token: str, correct_answers: list) -> bool:
    """Check if predicted token matches any accepted answer."""
    predicted_clean = predicted_token.replace("##", "").lower().strip()
    for ans in correct_answers:
        if ans.lower() in predicted_clean or predicted_clean in ans.lower():
            return True
    return False


def auroc(scores, labels):
    """Compute AUROC: P(score[correct] < score[wrong])."""
    pairs = [(s, l) for s, l in zip(scores, labels)]
    correct = [s for s, l in pairs if l == 1]
    wrong = [s for s, l in pairs if l == 0]
    if not correct or not wrong:
        return float("nan")
    n_pairs = len(correct) * len(wrong)
    concordant = sum(sc < sw for sc in correct for sw in wrong)
    tied = sum(sc == sw for sc in correct for sw in wrong)
    return (concordant + 0.5 * tied) / n_pairs


def bootstrap_auroc(scores, labels, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(scores)
    bootstrap_vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        bs_scores = [scores[i] for i in idx]
        bs_labels = [labels[i] for i in idx]
        a = auroc(bs_scores, bs_labels)
        if not math.isnan(a):
            bootstrap_vals.append(a)
    bootstrap_vals.sort()
    lo = bootstrap_vals[int(0.025 * len(bootstrap_vals))]
    hi = bootstrap_vals[int(0.975 * len(bootstrap_vals))]
    return lo, hi


def cohens_d(group_a, group_b):
    """Cohen's d (pooled std)."""
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return float("nan")
    ma, mb = np.mean(group_a), np.mean(group_b)
    sa, sb = np.std(group_a, ddof=1), np.std(group_b, ddof=1)
    sp = math.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    return abs(ma - mb) / sp


def run_pilot(K: int = 8, temperature: float = 1.0):
    tokenizer, model = load_distilbert()

    results = []
    t0 = time.time()

    for i, item in enumerate(QUESTIONS):
        q = item["q"]
        # DistilBERT uses [MASK] same as BERT
        tokens, token_ids = bpfc_single(q, tokenizer, model, K=K, temperature=temperature)

        if not tokens:
            print(f"  [{i+1:3d}] SKIP (no mask): {q[:50]}")
            continue

        sig2 = compute_sigma2(token_ids)
        maj = majority_token(tokens)
        correct = is_correct(maj, item["answers"])
        maj_conf = Counter(tokens).most_common(1)[0][1] / K

        results.append({
            "question": q,
            "tier": item["tier"],
            "answers": item["answers"],
            "sampled_tokens": tokens[:3],
            "majority": maj,
            "correct": correct,
            "sigma2_answer": sig2,
            "majority_conf": maj_conf,
        })

        if (i + 1) % 10 == 0:
            acc_so_far = sum(r["correct"] for r in results) / len(results)
            print(f"  [{i+1:3d}/{len(QUESTIONS)}] acc={acc_so_far:.2f} σ²={sig2:.4f} maj='{maj}' ({'✓' if correct else '✗'})")

    elapsed = time.time() - t0
    print(f"\nCompleted {len(results)} questions in {elapsed:.0f}s")

    # ─── Analysis ───
    correct_sigma2 = [r["sigma2_answer"] for r in results if r["correct"]]
    wrong_sigma2   = [r["sigma2_answer"] for r in results if not r["correct"]]
    sigma2_all     = [r["sigma2_answer"] for r in results]
    labels_all     = [int(r["correct"]) for r in results]
    maj_conf_all   = [r["majority_conf"] for r in results]

    accuracy = sum(labels_all) / len(labels_all)
    print(f"\nAccuracy: {accuracy:.3f} ({sum(labels_all)}/{len(labels_all)})")

    auroc_bpfc = auroc(sigma2_all, labels_all)
    auroc_maj  = auroc(maj_conf_all, [1 - l for l in labels_all])  # higher conf = more likely correct
    ci_lo, ci_hi = bootstrap_auroc(sigma2_all, labels_all)

    print(f"AUROC (σ²_answer): {auroc_bpfc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"AUROC (majority_conf): {auroc_maj:.3f}")

    d = cohens_d(correct_sigma2, wrong_sigma2)
    print(f"Cohen's d: {d:.3f}")

    mean_correct = np.mean(correct_sigma2) if correct_sigma2 else float("nan")
    mean_wrong   = np.mean(wrong_sigma2)   if wrong_sigma2   else float("nan")
    print(f"Mean σ² correct={mean_correct:.4f}, wrong={mean_wrong:.4f}")

    # Tier breakdown
    tier_stats = {}
    for tier in ["easy", "medium", "hard"]:
        tier_items = [r for r in results if r["tier"] == tier]
        if tier_items:
            tacc = sum(r["correct"] for r in tier_items) / len(tier_items)
            ts2  = np.mean([r["sigma2_answer"] for r in tier_items])
            tier_stats[tier] = {"acc": tacc, "mean_sigma2": ts2, "n": len(tier_items)}
    print("\nTier breakdown:")
    for tier, s in tier_stats.items():
        print(f"  {tier:8s}: acc={s['acc']:.2f} σ²={s['mean_sigma2']:.4f} n={s['n']}")

    output = {
        "model": "distilbert-base-uncased",
        "params_M": 66,
        "methodology": "temperature_sampling_K8",
        "n": len(results),
        "k": K,
        "temperature": temperature,
        "accuracy": accuracy,
        "elapsed_s": round(elapsed, 1),
        "bpfc": {
            "auroc_mean": auroc_bpfc,
            "auroc_ci_lo": ci_lo,
            "auroc_ci_hi": ci_hi,
        },
        "majority_conf": {
            "auroc_mean": auroc_maj,
        },
        "cohens_d": d,
        "mean_sigma2_correct": mean_correct,
        "mean_sigma2_wrong": mean_wrong,
        "tier_breakdown": tier_stats,
        "comparison": {
            "distilbert_auroc": auroc_bpfc,
            "bert_auroc": 0.791,
            "roberta_auroc": 0.642,
        },
    }

    out_path = "/home/azureuser/research/diffusion-meta-cognition-research-dr-claw/results/distilbert_crossval_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return output


if __name__ == "__main__":
    print("=" * 60)
    print("BPFC DistilBERT Cross-Architecture Validation")
    print("=" * 60)
    result = run_pilot(K=8, temperature=1.0)
    print("\n✅ DistilBERT cross-val complete")
    print(f"   AUROC={result['bpfc']['auroc_mean']:.3f} [{result['bpfc']['auroc_ci_lo']:.3f}, {result['bpfc']['auroc_ci_hi']:.3f}]")
    print(f"   Cohen's d={result['cohens_d']:.3f}, acc={result['accuracy']:.3f}")
