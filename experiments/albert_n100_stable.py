#!/usr/bin/env python3
"""
albert_n100_stable.py — N=100 ALBERT-large-v2 BPFC experiment for tighter CI.

Goal: Pool with session-17 N=50 result (AUROC=0.946) to get stable pooled estimate.
Protocol: same stratified bank (40 easy / 30 medium / 30 hard), K=8, temp=1.0, NTR metric.
Expected runtime: ~50s CPU.

Saves: results/albert_n100_stable_results.json
"""

import numpy as np
import json, time, random
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

SEED = 99
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_NAME = "albert-large-v2"
K = 8
TEMPERATURE = 1.0
N_EASY, N_MED, N_HARD = 40, 30, 30

# ── Question bank (100 stratified questions) ─────────────────────────────────
EASY_QUESTIONS = [
    ("The capital of France is [MASK].", "paris"),
    ("The chemical formula for water is [MASK].", "h2o"),
    ("William Shakespeare wrote the play [MASK].", "hamlet"),
    ("The planet closest to the Sun is [MASK].", "mercury"),
    ("The tallest mountain on Earth is [MASK].", "everest"),
    ("The currency of Japan is the [MASK].", "yen"),
    ("Albert Einstein developed the theory of [MASK].", "relativity"),
    ("The capital of Germany is [MASK].", "berlin"),
    ("The largest ocean on Earth is the [MASK] Ocean.", "pacific"),
    ("The primary language of Brazil is [MASK].", "portuguese"),
    ("The capital of Italy is [MASK].", "rome"),
    ("The chemical symbol for gold is [MASK].", "au"),
    ("The Great Wall was built in [MASK].", "china"),
    ("The capital of Australia is [MASK].", "canberra"),
    ("The speed of light is approximately 300,000 km per [MASK].", "second"),
    ("Isaac Newton discovered the law of [MASK].", "gravity"),
    ("The smallest continent is [MASK].", "australia"),
    ("The capital of Spain is [MASK].", "madrid"),
    ("The Amazon River flows through [MASK].", "brazil"),
    ("Beethoven composed his [MASK] Symphony when deaf.", "ninth"),
    ("The capital of Canada is [MASK].", "ottawa"),
    ("The chemical symbol for iron is [MASK].", "fe"),
    ("The Eiffel Tower is located in [MASK].", "paris"),
    ("The Declaration of Independence was signed in [MASK].", "1776"),
    ("The capital of Japan is [MASK].", "tokyo"),
    ("The Nile River is the longest river in [MASK].", "africa"),
    ("The human body has [MASK] bones.", "206"),
    ("The capital of China is [MASK].", "beijing"),
    ("Shakespeare was born in [MASK].", "stratford"),
    ("The first US President was George [MASK].", "washington"),
    ("The capital of Russia is [MASK].", "moscow"),
    ("The boiling point of water is [MASK] degrees Celsius.", "100"),
    ("Da Vinci painted the [MASK] Lisa.", "mona"),
    ("The capital of the UK is [MASK].", "london"),
    ("The periodic table element Au represents [MASK].", "gold"),
    ("The sun rises in the [MASK].", "east"),
    ("The capital of India is New [MASK].", "delhi"),
    ("Gravity was described by Isaac [MASK].", "newton"),
    ("The capital of Egypt is [MASK].", "cairo"),
    ("The French Revolution began in [MASK].", "1789"),
]

MEDIUM_QUESTIONS = [
    ("The author of 'Don Quixote' is Miguel de [MASK].", "cervantes"),
    ("The 1969 Moon landing mission was called Apollo [MASK].", "11"),
    ("The smallest planet in the solar system is [MASK].", "mercury"),
    ("DNA stands for deoxyribonucleic [MASK].", "acid"),
    ("The composer of 'The Four Seasons' is Antonio [MASK].", "vivaldi"),
    ("The treaty ending World War I was signed at [MASK].", "versailles"),
    ("The atomic number of carbon is [MASK].", "6"),
    ("The author of 'Crime and Punishment' is [MASK].", "dostoevsky"),
    ("The speed of sound at sea level is approximately [MASK] m/s.", "343"),
    ("The longest-reigning British monarch was Queen [MASK].", "victoria"),
    ("Photosynthesis converts CO2 and water into glucose and [MASK].", "oxygen"),
    ("The capital of Argentina is Buenos [MASK].", "aires"),
    ("The element with atomic symbol Na is [MASK].", "sodium"),
    ("The composer of Symphony No. 40 is Wolfgang Amadeus [MASK].", "mozart"),
    ("The Battle of Waterloo was in [MASK].", "1815"),
    ("The first antibiotic discovered was [MASK].", "penicillin"),
    ("The capital of South Africa is [MASK].", "pretoria"),
    ("The mathematician who proved Fermat's Last Theorem is Andrew [MASK].", "wiles"),
    ("The currency of Switzerland is the Swiss [MASK].", "franc"),
    ("Julius Caesar was assassinated in [MASK] BC.", "44"),
    ("The human genome has approximately [MASK] billion base pairs.", "3"),
    ("The Pythagorean theorem: a² + b² = [MASK].", "c²"),
    ("The author of 'The Divine Comedy' is Dante [MASK].", "alighieri"),
    ("The capital of Brazil is [MASK].", "brasilia"),
    ("The first programmable electronic computer was called [MASK].", "eniac"),
    ("The Battle of Hastings was in [MASK].", "1066"),
    ("The author of 'War and Peace' is Leo [MASK].", "tolstoy"),
    ("Archimedes' principle relates to buoyancy of objects in [MASK].", "fluid"),
    ("The chemical formula for table salt is [MASK].", "nacl"),
    ("The year the Berlin Wall fell was [MASK].", "1989"),
]

HARD_QUESTIONS = [
    ("The Treaty of [MASK] ended the Crimean War.", "paris"),
    ("The composer of 'Tristan und Isolde' is Richard [MASK].", "wagner"),
    ("The half-life of Carbon-14 is approximately [MASK] years.", "5730"),
    ("The philosopher who wrote 'Being and Time' is Martin [MASK].", "heidegger"),
    ("The largest moon of Neptune is [MASK].", "triton"),
    ("The author of 'The Sound and the Fury' is William [MASK].", "faulkner"),
    ("The capital of Kazakhstan is [MASK].", "astana"),
    ("The mathematician who developed non-Euclidean geometry is [MASK].", "riemann"),
    ("The Meiji Restoration occurred in [MASK].", "1868"),
    ("The chemist who discovered radioactivity is Henri [MASK].", "becquerel"),
    ("The capital of Burkina Faso is [MASK].", "ouagadougou"),
    ("The philosopher of the categorical imperative is Immanuel [MASK].", "kant"),
    ("The first antiparticle discovered was the [MASK].", "positron"),
    ("The author of 'The Trial' is Franz [MASK].", "kafka"),
    ("The country that uses the lempira as currency is [MASK].", "honduras"),
    ("The Battle of [MASK] was the turning point of WWI on the Eastern Front.", "tannenberg"),
    ("The composer of 'Winterreise' is Franz [MASK].", "schubert"),
    ("The capital of Kyrgyzstan is [MASK].", "bishkek"),
    ("The physicist who formulated wave mechanics is Erwin [MASK].", "schrodinger"),
    ("The element with atomic number 77 is [MASK].", "iridium"),
    ("The author of 'The Tin Drum' is Günter [MASK].", "grass"),
    ("The Ottoman sultan who conquered Constantinople was [MASK] II.", "mehmed"),
    ("The mathematician who created set theory is Georg [MASK].", "cantor"),
    ("The capital of Tajikistan is [MASK].", "dushanbe"),
    ("The philosopher who wrote 'Thus Spoke Zarathustra' is Friedrich [MASK].", "nietzsche"),
    ("The supernova remnant from 1054 AD is the [MASK] Nebula.", "crab"),
    ("The painter of 'The Ambassadors' (1533) is Hans [MASK].", "holbein"),
    ("The capital of Eritrea is [MASK].", "asmara"),
    ("The isotope used in MRI scanners is hydrogen-[MASK].", "1"),
    ("The economist who wrote 'The General Theory' is John Maynard [MASK].", "keynes"),
]

print(f"ALBERT-large-v2 N=100 stability experiment (K={K}, temp={TEMPERATURE})")
print(f"Questions: {N_EASY} easy / {N_MED} medium / {N_HARD} hard")

# ── Load model ────────────────────────────────────────────────────────────────
t0 = time.time()
print("Loading ALBERT-large-v2...")
tokenizer = AutoTokenizer.from_pretrained("albert-large-v2")
model = AutoModelForMaskedLM.from_pretrained("albert-large-v2")
model.eval()
MASK_TOKEN = tokenizer.mask_token  # '<mask>'
print(f"Loaded in {time.time()-t0:.1f}s")

# ── NTR uncertainty metric ────────────────────────────────────────────────────
def compute_ntr(template: str, k: int = K, temperature: float = TEMPERATURE) -> tuple[float, str]:
    """
    Returns (ntr, majority_pred).
    NTR = Normalized Type-Token Ratio = len(unique_sampled_tokens) / k
    """
    # Ensure mask token matches the tokenizer
    if "[MASK]" in template:
        template = template.replace("[MASK]", MASK_TOKEN)
    
    inputs = tokenizer(template, return_tensors="pt")
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    
    if len(mask_positions) == 0:
        return 0.5, "?"
    
    mask_pos = mask_positions[0].item()
    
    sampled_tokens = []
    with torch.no_grad():
        logits = model(**inputs).logits  # [1, seq_len, vocab_size]
    
    mask_logits = logits[0, mask_pos, :]  # [vocab_size]
    probs = torch.softmax(mask_logits / temperature, dim=-1)
    
    for _ in range(k):
        token_id = torch.multinomial(probs, 1).item()
        sampled_tokens.append(tokenizer.decode([token_id]).strip().lower())
    
    ntr = len(set(sampled_tokens)) / k
    majority = max(set(sampled_tokens), key=sampled_tokens.count)
    return ntr, majority

# ── Run experiment ────────────────────────────────────────────────────────────
all_questions = (
    [(q, a, "easy") for q, a in EASY_QUESTIONS[:N_EASY]] +
    [(q, a, "medium") for q, a in MEDIUM_QUESTIONS[:N_MED]] +
    [(q, a, "hard") for q, a in HARD_QUESTIONS[:N_HARD]]
)
random.shuffle(all_questions)

results = []
t_exp = time.time()
for i, (template, gold, tier) in enumerate(all_questions):
    ntr, pred = compute_ntr(template)
    correct = int(gold.lower() in pred.lower() or pred.lower() in gold.lower())
    results.append({
        "template": template,
        "gold": gold,
        "pred": pred,
        "ntr": ntr,
        "correct": correct,
        "tier": tier,
    })
    if (i+1) % 20 == 0:
        elapsed = time.time() - t_exp
        print(f"  {i+1}/{len(all_questions)} completed ({elapsed:.1f}s elapsed)")

total_time = time.time() - t_exp
print(f"Experiment done in {total_time:.1f}s")

# ── AUROC computation ─────────────────────────────────────────────────────────
def bootstrap_auroc(scores, labels, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(scores)
    scores, labels = np.array(scores), np.array(labels)
    
    def auroc(s, l):
        n_pos = l.sum(); n_neg = (1-l).sum()
        if n_pos == 0 or n_neg == 0: return 0.5
        ranks = np.argsort(np.argsort(-s))
        pos_ranks = ranks[l == 1]
        return (pos_ranks.sum() - n_pos*(n_pos-1)/2) / (n_pos * n_neg)
    
    obs = auroc(scores, labels)
    boots = [auroc(scores[idx:=rng.integers(0,n,n)], labels[idx]) for _ in range(n_boot)]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return obs, ci_lo, ci_hi

ntrs = [r["ntr"] for r in results]
corrects = [r["correct"] for r in results]

auroc, lo, hi = bootstrap_auroc(ntrs, corrects)
acc = np.mean(corrects)
print(f"\nOverall (N={len(results)}, K={K}):")
print(f"  AUROC = {auroc:.3f} [{lo:.3f}, {hi:.3f}]")
print(f"  Accuracy = {acc:.3f}")

# Cohen's d
ntr_arr = np.array(ntrs); corr_arr = np.array(corrects)
correct_ntrs = ntr_arr[corr_arr == 1]
wrong_ntrs = ntr_arr[corr_arr == 0]
if len(correct_ntrs) > 1 and len(wrong_ntrs) > 1:
    pooled_std = np.sqrt(((len(correct_ntrs)-1)*correct_ntrs.std()**2 + (len(wrong_ntrs)-1)*wrong_ntrs.std()**2) / (len(correct_ntrs)+len(wrong_ntrs)-2))
    cohens_d = (wrong_ntrs.mean() - correct_ntrs.mean()) / (pooled_std + 1e-9)
    print(f"  Cohen's d = {cohens_d:.3f}")
else:
    cohens_d = float('nan')

# Per-tier breakdown
for tier in ["easy", "medium", "hard"]:
    tier_r = [r for r in results if r["tier"] == tier]
    if not tier_r: continue
    t_ntrs = [r["ntr"] for r in tier_r]
    t_corr = [r["correct"] for r in tier_r]
    t_auroc, t_lo, t_hi = bootstrap_auroc(t_ntrs, t_corr)
    print(f"  {tier.capitalize()} (N={len(tier_r)}): AUROC={t_auroc:.3f} [{t_lo:.3f}, {t_hi:.3f}], acc={np.mean(t_corr):.2f}")

# ── Pooled estimate with Session 17 ──────────────────────────────────────────
print("\n── Pooled estimate (Session 17 N=50 AUROC=0.946 + this run N=100) ──")
# Weighted average by N
n17, auroc17 = 50, 0.946
n_this = len(results)
pooled_auroc = (n17 * auroc17 + n_this * auroc) / (n17 + n_this)
print(f"  Pooled AUROC (N=150) ≈ {pooled_auroc:.3f}")

# ── Save results ──────────────────────────────────────────────────────────────
import os
os.makedirs("results", exist_ok=True)
output = {
    "experiment": "albert_n100_stable",
    "model": MODEL_NAME,
    "n": len(results),
    "k": K,
    "temperature": TEMPERATURE,
    "auroc": auroc,
    "ci_lo": lo,
    "ci_hi": hi,
    "cohens_d": cohens_d,
    "accuracy": acc,
    "runtime_s": total_time,
    "pooled_auroc_with_session17": pooled_auroc,
    "pooled_n": n17 + n_this,
    "results": results,
}
out_path = "results/albert_n100_stable_results.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n✅ Saved → {out_path}")
