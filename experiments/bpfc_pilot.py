#!/usr/bin/env python3
"""
BPFC Pilot Experiment: Bayesian Posterior Factual Calibration
=============================================================
GPU-Free design — uses HF Space Gradio API (multimodalart/LLaDA) 
for K=8 independent denoising passes per question.

Theoretical basis: Doyle (arXiv:2507.07586) proves that K independent
masked denoising passes converge to the exact Bayesian posterior at rate
O(1/√K). We operationalize this as answer-level behavioral variance:
    - variance(answers) ≈ epistemic uncertainty
    - high variance → model doesn't "know" → likely incorrect

DESIGN:
  - Dataset: TriviaQA dev set (first N=50 questions, 1-hop factual)
  - Model: LLaDA-8B-Instruct via HF Space (Gradio API, free, ZeroGPU)
  - K=8 independent passes per question (K chosen per Doyle Sec.4, ρ=0.996 at K≥16; K=8 is budget compromise)
  - Variance metric: σ²_answer = 1 - mean_pairwise_agreement (lexical)
  - Calibration metric: AUROC(σ²_answer, 1-correct)
  - AR baseline: GPT-4o-mini semantic entropy (K=8 samples, same questions)

COST ESTIMATE:
  - HF Space calls: free (ZeroGPU community grant)
  - GPT-4o-mini AR baseline: ~N*K*150tokens = 50*8*150 = 60K tokens ≈ $0.009
  - Total: < $0.05

RUNTIME ESTIMATE (sequential API calls):
  - 50 questions × 8 passes × 30s/pass = ~200 min (parallelize → 25 min)
  - Use async/concurrent calls, max 3 concurrent to respect rate limits

OUTPUTS:
  - data/bpfc_pilot_results.jsonl  — per-question results
  - data/bpfc_pilot_analysis.json  — aggregate metrics (AUROC, ECE, etc.)
  - figures/bpfc_calibration_curve.png (optional if matplotlib available)

Author: Dr. Claw | 2026-02-27
"""

import json
import time
import os
import random
import hashlib
import asyncio
import statistics
import unicodedata
import string
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "N_QUESTIONS": 50,           # TriviaQA questions
    "K_PASSES": 8,               # Independent denoising samples per question
    "MAX_CONCURRENT": 3,         # Max parallel API calls (HF Space rate limit)
    "ANSWER_MAX_TOKENS": 50,     # Max tokens for answer
    "QUESTION_MAX_TOKENS": 128,  # Max prompt tokens
    "SEED": 42,
    "OUTPUT_DIR": Path(__file__).parent.parent / "data",
    "RESULTS_FILE": "bpfc_pilot_results.jsonl",
    "ANALYSIS_FILE": "bpfc_pilot_analysis.json",
    "HF_SPACE_NAME": "multimodalart/LLaDA",   # Primary LLaDA endpoint
    "HF_SPACE_ALT": "spuun/llada-8b-kcv",    # Fallback space
    "OPENAI_MODEL": "gpt-4o-mini",            # AR baseline
    "TRIVIAQA_CACHE": "data/triviaqa_sample.jsonl",  # local cache if downloaded
}

CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)

# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    question_id: str
    question: str
    gold_answers: list
    # BPFC fields
    dlm_answers: list        # K=8 sampled answers from LLaDA
    dlm_agreement: float     # pairwise agreement rate
    dlm_sigma2: float        # 1 - agreement (our σ²_answer proxy)
    dlm_correct: bool        # any answer matches gold
    # AR baseline fields
    ar_answers: list         # K=8 sampled answers from GPT-4o-mini
    ar_agreement: float
    ar_sigma2: float
    ar_correct: bool
    # Metadata
    timestamp: float
    llada_api_errors: int
    openai_api_errors: int


# ─── TriviaQA Loader ──────────────────────────────────────────────────────────

def load_triviaqa_sample(n: int = 50, seed: int = 42) -> list[dict]:
    """
    Load N TriviaQA dev questions. Tries:
    1. Local cache
    2. HuggingFace datasets (datasets library)
    3. Manual download of dev JSON
    Returns list of {question_id, question, answers: [str]} dicts.
    """
    cache_path = Path(CONFIG["TRIVIAQA_CACHE"])
    
    # Try local cache first
    if cache_path.exists():
        print(f"[TriviaQA] Loading from cache: {cache_path}")
        questions = []
        with open(cache_path) as f:
            for line in f:
                questions.append(json.loads(line))
        random.seed(seed)
        return random.sample(questions, min(n, len(questions)))
    
    # Try HuggingFace datasets
    try:
        from datasets import load_dataset
        print("[TriviaQA] Downloading from HuggingFace datasets...")
        ds = load_dataset("trivia_qa", "rc", split="validation", streaming=False)
        
        questions = []
        seen_ids = set()
        for item in ds:
            if len(questions) >= n * 3:  # Get 3x for diversity
                break
            qid = item["question_id"]
            if qid in seen_ids:
                continue
            seen_ids.add(qid)
            
            # Normalize answers
            answers = []
            if "answer" in item:
                ans = item["answer"]
                if "normalized_aliases" in ans:
                    answers.extend(ans["normalized_aliases"])
                elif "value" in ans:
                    answers.append(ans["value"])
            
            if not answers:
                continue
                
            questions.append({
                "question_id": qid,
                "question": item["question"],
                "answers": answers[:10]  # Keep top 10 aliases
            })
        
        # Sample n
        random.seed(seed)
        sampled = random.sample(questions, min(n, len(questions)))
        
        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            for q in sampled:
                f.write(json.dumps(q) + "\n")
        
        print(f"[TriviaQA] Loaded {len(sampled)} questions, saved cache.")
        return sampled
        
    except Exception as e:
        print(f"[TriviaQA] HuggingFace datasets failed: {e}")
        print("[TriviaQA] Using hardcoded fallback questions...")
        return _triviaqa_fallback(n, seed)


def _triviaqa_fallback(n: int, seed: int) -> list[dict]:
    """15 hardcoded TriviaQA-style questions for quick testing without datasets."""
    items = [
        {"question_id": "tqa_001", "question": "Who wrote the play 'Hamlet'?",
         "answers": ["william shakespeare", "shakespeare"]},
        {"question_id": "tqa_002", "question": "What is the chemical symbol for gold?",
         "answers": ["au"]},
        {"question_id": "tqa_003", "question": "In which year did World War II end?",
         "answers": ["1945"]},
        {"question_id": "tqa_004", "question": "What is the capital of Australia?",
         "answers": ["canberra"]},
        {"question_id": "tqa_005", "question": "Who painted the Mona Lisa?",
         "answers": ["leonardo da vinci", "da vinci", "leonardo"]},
        {"question_id": "tqa_006", "question": "What is the smallest planet in our solar system?",
         "answers": ["mercury"]},
        {"question_id": "tqa_007", "question": "Who invented the telephone?",
         "answers": ["alexander graham bell", "graham bell"]},
        {"question_id": "tqa_008", "question": "What is the longest river in the world?",
         "answers": ["nile", "nile river"]},
        {"question_id": "tqa_009", "question": "In what year was the Eiffel Tower completed?",
         "answers": ["1889"]},
        {"question_id": "tqa_010", "question": "Who was the first person to walk on the moon?",
         "answers": ["neil armstrong", "armstrong"]},
        {"question_id": "tqa_011", "question": "What is the chemical formula for water?",
         "answers": ["h2o", "h₂o"]},
        {"question_id": "tqa_012", "question": "Who wrote '1984'?",
         "answers": ["george orwell", "orwell", "eric arthur blair"]},
        {"question_id": "tqa_013", "question": "What is the speed of light in a vacuum (approx)?",
         "answers": ["299,792,458 m/s", "3×10^8 m/s", "300000 km/s", "186000 miles per second"]},
        {"question_id": "tqa_014", "question": "Which element has the atomic number 79?",
         "answers": ["gold", "au"]},
        {"question_id": "tqa_015", "question": "Who composed 'The Four Seasons'?",
         "answers": ["antonio vivaldi", "vivaldi"]},
        {"question_id": "tqa_016", "question": "What is the tallest mountain in the world?",
         "answers": ["mount everest", "everest", "mt everest"]},
        {"question_id": "tqa_017", "question": "In which country is the Amazon River primarily located?",
         "answers": ["brazil"]},
        {"question_id": "tqa_018", "question": "Who developed the theory of general relativity?",
         "answers": ["albert einstein", "einstein"]},
        {"question_id": "tqa_019", "question": "What is the capital of Japan?",
         "answers": ["tokyo"]},
        {"question_id": "tqa_020", "question": "What year did the Titanic sink?",
         "answers": ["1912"]},
    ]
    random.seed(seed)
    return random.sample(items, min(n, len(items)))


# ─── Answer Normalization ──────────────────────────────────────────────────────

def normalize_answer(text: str) -> str:
    """Standard TriviaQA normalization: lowercase, remove articles/punctuation."""
    def remove_articles(t):
        return ' '.join(w for w in t.split() if w not in {'a', 'an', 'the'})
    
    def white_space_fix(t):
        return ' '.join(t.split())
    
    def remove_punc(t):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in t if ch not in exclude)
    
    def lower(t):
        return t.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(text))))


def answer_is_correct(predicted: str, gold_answers: list) -> bool:
    """Check if predicted answer matches any gold answer (after normalization)."""
    pred_norm = normalize_answer(predicted)
    return any(normalize_answer(g) in pred_norm or pred_norm in normalize_answer(g)
               for g in gold_answers)


def pairwise_agreement(answers: list[str]) -> float:
    """
    Compute mean pairwise exact-match agreement among K answers.
    This is our proxy for 1 - σ²_answer (higher = more confident).
    
    A more principled version would use sentence-transformer cosine similarity,
    but lexical agreement is interpretable and fast (CPU-only).
    """
    if len(answers) < 2:
        return 1.0
    
    n = len(answers)
    norm_answers = [normalize_answer(a) for a in answers]
    
    # Count pairwise agreements
    agreements = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            # Partial match: one is a substring of the other
            if (norm_answers[i] in norm_answers[j] or 
                norm_answers[j] in norm_answers[i] or
                norm_answers[i] == norm_answers[j]):
                agreements += 1
    
    return agreements / pairs if pairs > 0 else 1.0


# ─── LLaDA API Client (Gradio Space) ─────────────────────────────────────────

class LLaDASpaceClient:
    """
    Client for LLaDA-8B-Instruct via HuggingFace Gradio Space.
    
    Strategy (in order of preference):
    1. Pure urllib Gradio API (no dependencies) — tries multimodalart/LLaDA space
    2. gradio_client library (if installed: pip install gradio_client)
    
    CONFIRMED Feb 2026: LLaDA-8B-Instruct has NO HF Inference Providers.
    All variants return HTTP 410 on api-inference.huggingface.co.
    The ZeroGPU Space (multimodalart/LLaDA) is the only free cloud access route.
    
    NOTE on outputs: The Space returns a text string (final generation).
    We do NOT have access to per-step logits or token-level probabilities.
    Our K=8 independent samples are the operationalization of Doyle's
    K independent denoising passes at the *behavioral* level.
    
    LIMITATION: This captures answer-level variance, not token-level σ².
    This is a valid proxy for the pilot (confirms signal exists), but
    the full paper will need direct model access for per-token σ².
    
    GRADIO API STRUCTURE for multimodalart/LLaDA:
    - Space URL: https://multimodalart-llada.hf.space
    - API info: GET /info
    - Predict: POST /run/predict  (fn_index=0 for main generation)
    - Session: GET /queue/join → SSE stream
    """
    
    # HF Space endpoints for LLaDA
    SPACE_URLS = [
        "https://multimodalart-llada.hf.space",
        "https://spuun-llada-8b-kcv.hf.space",
        "https://ginigen-llada.hf.space",
    ]
    
    def __init__(self, space_name: str = "multimodalart/LLaDA"):
        self.space_name = space_name
        self.space_url = self._space_name_to_url(space_name)
        self.available = False
        self.use_gradio_client = False
        self._probe_availability()
    
    def _space_name_to_url(self, name: str) -> str:
        """Convert 'user/SpaceName' → 'https://user-spacename.hf.space'"""
        parts = name.lower().replace("/", "-")
        return f"https://{parts}.hf.space"
    
    def _probe_availability(self):
        """Check which Space endpoints are reachable."""
        import urllib.request
        import urllib.error
        
        # Try primary space URL
        urls_to_try = [self.space_url] + self.SPACE_URLS
        
        for url in urls_to_try:
            try:
                req = urllib.request.Request(f"{url}/info", 
                    headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        self.space_url = url
                        self.available = True
                        print(f"[LLaDA] Space available: {url}")
                        return
            except Exception:
                continue
        
        # Try gradio_client as fallback
        try:
            from gradio_client import Client
            self.gradio_client = Client(self.space_name)
            self.available = True
            self.use_gradio_client = True
            print(f"[LLaDA] Connected via gradio_client: {self.space_name}")
        except ImportError:
            print("[LLaDA] gradio_client not installed. Run: pip install gradio_client")
        except Exception as e:
            print(f"[LLaDA] All connection methods failed. Error: {e}")
        
        if not self.available:
            print("[LLaDA] WARNING: No API access. Will use dry-run mode.")
    
    def _format_prompt(self, question: str) -> str:
        """Format question as LLaDA-8B-Instruct chat prompt."""
        # LLaDA-8B-Instruct uses Llama-3 chat template
        return (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Answer the following question with a short, direct answer. "
            f"Give only the answer, no explanation.\n\n"
            f"Question: {question}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    
    def _generate_via_urllib(self, question: str) -> Optional[str]:
        """
        Call LLaDA Space via pure urllib (Gradio REST API).
        
        Gradio Spaces expose: POST /run/predict with JSON body:
        {"data": [prompt, max_new_tokens, steps, ...], "fn_index": 0}
        
        The exact fn_index and data format depends on the Space's app.py.
        For multimodalart/LLaDA the typical inputs are:
        - message (str)
        - history (list, empty for fresh)
        - gen_length (int, default 128)
        - steps (int, default 32)
        - word_constraints (str, empty)
        """
        import urllib.request
        import urllib.error
        
        # Gradio API predict endpoint
        url = f"{self.space_url}/run/predict"
        
        payload = json.dumps({
            "data": [
                question,     # message / prompt
                [],           # history (empty = fresh)
                64,           # gen_length (short answers)
                32,           # steps
                ""            # word_constraints (none)
            ],
            "fn_index": 0
        }).encode("utf-8")
        
        req = urllib.request.Request(url, data=payload,
            headers={"Content-Type": "application/json"})
        
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
                # Gradio response: {"data": [...], "duration": ...}
                if "data" in result and result["data"]:
                    output = result["data"][0]
                    # Could be str or list of chat messages
                    if isinstance(output, str):
                        return output.strip()
                    elif isinstance(output, list) and output:
                        # Chat history format: [[user, bot], ...]
                        last_pair = output[-1]
                        if isinstance(last_pair, list) and len(last_pair) > 1:
                            return str(last_pair[1]).strip()
                    return str(output).strip()
        except urllib.error.HTTPError as e:
            print(f"[LLaDA] HTTP {e.code}: {e.reason}")
        except Exception as e:
            print(f"[LLaDA] urllib error: {e}")
        
        return None
    
    def _generate_via_gradio_client(self, question: str) -> Optional[str]:
        """Call via gradio_client library."""
        try:
            result = self.gradio_client.predict(
                message=question,
                api_name="/chat"
            )
            if isinstance(result, (list, tuple)):
                result = result[0] if result else ""
            return str(result).strip()
        except Exception as e:
            print(f"[LLaDA] gradio_client error: {e}")
            return None
    
    def generate_answer(self, question: str, max_retries: int = 3) -> Optional[str]:
        """Run one denoising pass and return the answer."""
        if not self.available:
            return None
        
        for attempt in range(max_retries):
            try:
                if self.use_gradio_client:
                    result = self._generate_via_gradio_client(question)
                else:
                    result = self._generate_via_urllib(question)
                
                if result is not None:
                    return result
                    
            except Exception as e:
                pass
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        
        print(f"[LLaDA] Generate failed after {max_retries} retries")
        return None
    
    def sample_k_answers(self, question: str, k: int = 8) -> tuple[list[str], int]:
        """
        Run K independent denoising passes for one question.
        Returns (answers, num_errors).
        
        IMPORTANT: Each call is a fresh, independent denoising pass.
        LLaDA is stochastic by default (random mask sampling at each step).
        K independent calls = K independent samples from the posterior.
        This is exactly what Doyle (2025) Eq.6 describes as the MC estimator.
        """
        answers = []
        errors = 0
        
        for i in range(k):
            answer = self.generate_answer(question)
            if answer is not None:
                answers.append(answer)
            else:
                errors += 1
            # Rate limiting: 1s between calls (ZeroGPU community grant, be polite)
            time.sleep(1.5)
        
        return answers, errors


# ─── OpenAI AR Baseline ───────────────────────────────────────────────────────

class OpenAIARBaseline:
    """
    GPT-4o-mini baseline using temperature=0.8 sampling to simulate
    K independent draws. This mirrors Kuhn et al. (2023) semantic entropy
    but with lexical agreement instead of NLI clustering.
    
    Cost: ~$0.009 for 50 questions × 8 samples × 150 tokens
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = None
        self._init_client()
    
    def _init_client(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[OpenAI] OPENAI_API_KEY not set. AR baseline disabled.")
            return
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            print(f"[OpenAI] AR baseline ready: {self.model}")
        except ImportError:
            print("[OpenAI] openai not installed. Run: pip install openai")
    
    def sample_k_answers(self, question: str, k: int = 8) -> tuple[list[str], int]:
        """Sample K answers with temperature=0.8."""
        if self.client is None:
            return [], k
        
        answers = []
        errors = 0
        prompt = f"Answer the following question with a short, direct answer. Give only the answer.\n\nQuestion: {question}"
        
        try:
            # Batch K samples in one API call using n=k
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                n=k,
                temperature=0.8,
                max_tokens=50
            )
            answers = [choice.message.content.strip() for choice in response.choices]
        except Exception as e:
            print(f"[OpenAI] API error: {e}")
            errors = k
        
        return answers, errors


# ─── Calibration Metrics ──────────────────────────────────────────────────────

def compute_auroc(scores: list[float], labels: list[bool]) -> float:
    """
    Compute AUROC for uncertainty (σ²) predicting error (1-correct).
    Higher σ² should predict incorrect answers (labels=True for errors).
    
    Uses numpy-free manual AUROC computation (rank-based).
    """
    paired = sorted(zip(scores, labels), reverse=True)  # sort by score descending
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined
    
    # Trapezoidal AUROC via rank sum
    # This is equivalent to Mann-Whitney U statistic
    tp = 0
    fp = 0
    auc = 0.0
    prev_fp = 0
    
    for score, label in paired:
        if label:  # True positive (uncertain + incorrect)
            tp += 1
        else:
            fp += 1
            auc += tp  # Count how many TPs came before this FP
    
    return auc / (n_pos * n_neg)


def compute_ece(sigma2_scores: list[float], correct_labels: list[bool], n_bins: int = 10) -> float:
    """
    Expected Calibration Error: bin by σ² (converted to confidence = 1-σ²),
    check if accuracy matches confidence within bins.
    """
    confidences = [1.0 - s for s in sigma2_scores]
    paired = sorted(zip(confidences, correct_labels))
    
    bin_size = len(paired) // n_bins
    if bin_size == 0:
        return float('nan')
    
    ece = 0.0
    for i in range(n_bins):
        bin_items = paired[i * bin_size: (i + 1) * bin_size]
        if not bin_items:
            continue
        bin_conf = statistics.mean(c for c, _ in bin_items)
        bin_acc = statistics.mean(float(l) for _, l in bin_items)
        ece += abs(bin_conf - bin_acc) * len(bin_items) / len(paired)
    
    return ece


# ─── Main Experiment Loop ─────────────────────────────────────────────────────

def run_pilot(
    n_questions: int = CONFIG["N_QUESTIONS"],
    k_passes: int = CONFIG["K_PASSES"],
    dry_run: bool = False,
    resume: bool = True,
) -> dict:
    """
    Main experiment loop. 
    
    Args:
        n_questions: Number of TriviaQA questions
        k_passes: Independent denoising passes per question
        dry_run: If True, skip API calls, use synthetic data for testing
        resume: If True, skip already-processed questions (load from results file)
    
    Returns:
        Analysis dict with AUROC, ECE, etc.
    """
    
    print("=" * 60)
    print("BPFC PILOT EXPERIMENT")
    print(f"N={n_questions} questions, K={k_passes} passes")
    print(f"Dry run: {dry_run}")
    print("=" * 60)
    
    # Initialize clients
    llada_client = LLaDASpaceClient() if not dry_run else None
    ar_client = OpenAIARBaseline() if not dry_run else None
    
    # Load questions
    questions = load_triviaqa_sample(n_questions)
    print(f"[Main] Loaded {len(questions)} questions")
    
    # Load existing results if resuming
    results_path = CONFIG["OUTPUT_DIR"] / CONFIG["RESULTS_FILE"]
    processed_ids = set()
    if resume and results_path.exists():
        with open(results_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    processed_ids.add(r["question_id"])
                except:
                    pass
        print(f"[Main] Resuming: {len(processed_ids)} already processed")
    
    # Process questions
    results = []
    with open(results_path, "a") as out_f:
        for idx, q in enumerate(questions):
            qid = q["question_id"]
            if qid in processed_ids:
                print(f"[{idx+1}/{len(questions)}] SKIP (already done): {qid}")
                continue
            
            print(f"\n[{idx+1}/{len(questions)}] Processing: {q['question'][:70]}...")
            
            if dry_run:
                # Synthetic data for testing pipeline
                dlm_answers = _synthetic_answers(q["answers"], k_passes)
                ar_answers = _synthetic_answers(q["answers"], k_passes)
                llada_errors = 0
                ar_errors = 0
            else:
                # Real API calls
                dlm_answers, llada_errors = llada_client.sample_k_answers(
                    q["question"], k=k_passes
                ) if llada_client.client else ([], k_passes)
                
                ar_answers, ar_errors = ar_client.sample_k_answers(
                    q["question"], k=k_passes
                ) if ar_client.client else ([], k_passes)
            
            # Compute metrics
            dlm_agreement = pairwise_agreement(dlm_answers) if dlm_answers else 0.0
            ar_agreement = pairwise_agreement(ar_answers) if ar_answers else 0.0
            
            dlm_correct = any(answer_is_correct(a, q["answers"]) for a in dlm_answers)
            ar_correct = any(answer_is_correct(a, q["answers"]) for a in ar_answers)
            
            result = {
                "question_id": qid,
                "question": q["question"],
                "gold_answers": q["answers"][:5],
                "dlm_answers": dlm_answers,
                "dlm_agreement": dlm_agreement,
                "dlm_sigma2": 1.0 - dlm_agreement,  # Our posterior variance proxy
                "dlm_correct": dlm_correct,
                "ar_answers": ar_answers,
                "ar_agreement": ar_agreement,
                "ar_sigma2": 1.0 - ar_agreement,
                "ar_correct": ar_correct,
                "timestamp": time.time(),
                "llada_api_errors": llada_errors,
                "openai_api_errors": ar_errors,
            }
            
            results.append(result)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            
            print(f"  DLM: agreement={dlm_agreement:.2f}, σ²={result['dlm_sigma2']:.2f}, correct={dlm_correct}")
            print(f"   AR: agreement={ar_agreement:.2f}, σ²={result['ar_sigma2']:.2f}, correct={ar_correct}")
    
    # Load all results for analysis (including previously processed)
    all_results = []
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                try:
                    all_results.append(json.loads(line))
                except:
                    pass
    
    # Analysis
    analysis = _analyze(all_results)
    
    # Save analysis
    analysis_path = CONFIG["OUTPUT_DIR"] / CONFIG["ANALYSIS_FILE"]
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    _print_results(analysis)
    
    return analysis


def _synthetic_answers(gold: list, k: int) -> list[str]:
    """Generate synthetic K answers for dry run (60% correct, 40% wrong)."""
    answers = []
    for i in range(k):
        if random.random() < 0.6 and gold:
            answers.append(random.choice(gold))
        else:
            distractors = ["paris", "london", "1776", "newton", "unknown", "n/a"]
            answers.append(random.choice(distractors))
    return answers


def _analyze(results: list[dict]) -> dict:
    """Compute aggregate calibration metrics."""
    if not results:
        return {"error": "No results"}
    
    dlm_sigma2 = [r["dlm_sigma2"] for r in results]
    dlm_correct = [r["dlm_correct"] for r in results]
    ar_sigma2 = [r["ar_sigma2"] for r in results]
    ar_correct = [r["ar_correct"] for r in results]
    
    # AUROC: σ² predicts error (1-correct)
    dlm_errors = [not c for c in dlm_correct]
    ar_errors = [not c for c in ar_correct]
    
    dlm_auroc = compute_auroc(dlm_sigma2, dlm_errors)
    ar_auroc = compute_auroc(ar_sigma2, ar_errors)
    
    # ECE
    dlm_ece = compute_ece(dlm_sigma2, dlm_correct)
    ar_ece = compute_ece(ar_sigma2, ar_correct)
    
    # Correlation between DLM and AR σ²
    if len(dlm_sigma2) > 2 and len(ar_sigma2) > 2:
        # Spearman rank correlation (numpy-free)
        dlm_ranks = _rank(dlm_sigma2)
        ar_ranks = _rank(ar_sigma2)
        n = len(dlm_sigma2)
        cov = sum((dlm_ranks[i] - statistics.mean(dlm_ranks)) * 
                  (ar_ranks[i] - statistics.mean(ar_ranks)) 
                  for i in range(n))
        dlm_std = statistics.stdev(dlm_ranks)
        ar_std = statistics.stdev(ar_ranks)
        spearman_rho = cov / (n * dlm_std * ar_std) if dlm_std > 0 and ar_std > 0 else 0.0
    else:
        spearman_rho = float('nan')
    
    return {
        "n_questions": len(results),
        "k_passes": results[0].get("dlm_answers", []) and len(results[0]["dlm_answers"]),
        "dlm": {
            "auroc": dlm_auroc,
            "ece": dlm_ece,
            "accuracy": statistics.mean(float(c) for c in dlm_correct),
            "mean_sigma2": statistics.mean(dlm_sigma2),
            "mean_agreement": statistics.mean(r["dlm_agreement"] for r in results),
            "api_error_rate": statistics.mean(r["llada_api_errors"] / CONFIG["K_PASSES"] 
                                               for r in results),
        },
        "ar": {
            "auroc": ar_auroc,
            "ece": ar_ece,
            "accuracy": statistics.mean(float(c) for c in ar_correct),
            "mean_sigma2": statistics.mean(ar_sigma2),
            "mean_agreement": statistics.mean(r["ar_agreement"] for r in results),
            "api_error_rate": statistics.mean(r["openai_api_errors"] / CONFIG["K_PASSES"]
                                               for r in results),
        },
        "comparison": {
            "dlm_vs_ar_auroc_delta": dlm_auroc - ar_auroc,
            "spearman_rho_sigma2_correlation": spearman_rho,
            "hypothesis_H1_supported": dlm_auroc >= 0.65,
            "hypothesis_H1_label": "σ²_answer AUROC ≥ 0.65 (better than chance)",
        },
        "decision_gate": _decision_gate(dlm_auroc),
    }


def _decision_gate(auroc: float) -> dict:
    """
    From the BPFC research plan:
    - AUROC < 0.60: Reassess direction
    - AUROC 0.60-0.65: Marginal, revisit variance metric
    - AUROC ≥ 0.65: PROCEED to full study (N=500, token-level σ²)
    - AUROC ≥ 0.75: Strong signal, prioritize submission
    """
    if auroc < 0.60:
        decision = "REASSESS"
        action = "Signal too weak. Fall back to CZEC (confusion zone entropy) direction."
    elif auroc < 0.65:
        decision = "MARGINAL"
        action = "Try embedding-based similarity instead of lexical agreement. Increase K."
    elif auroc < 0.75:
        decision = "PROCEED"
        action = "Proceed to full N=500 study. Get direct model access for token-level σ²."
    else:
        decision = "STRONG_SIGNAL"
        action = "Prioritize submission. Write full paper. Token-level σ² likely even better."
    
    return {"auroc": auroc, "decision": decision, "action": action}


def _rank(scores: list[float]) -> list[float]:
    """Compute ranks for Spearman correlation."""
    sorted_scores = sorted(enumerate(scores), key=lambda x: x[1])
    ranks = [0.0] * len(scores)
    for rank, (idx, _) in enumerate(sorted_scores):
        ranks[idx] = rank + 1.0
    return ranks


def _print_results(analysis: dict):
    """Pretty-print analysis results."""
    print(f"N = {analysis.get('n_questions', '?')} questions")
    print()
    
    dlm = analysis.get("dlm", {})
    ar = analysis.get("ar", {})
    
    print("                    DLM (LLaDA)    AR (GPT-4o-mini)")
    print(f"  Accuracy:         {dlm.get('accuracy', 0):.3f}          {ar.get('accuracy', 0):.3f}")
    print(f"  Mean σ²:          {dlm.get('mean_sigma2', 0):.3f}          {ar.get('mean_sigma2', 0):.3f}")
    print(f"  AUROC(σ²→error):  {dlm.get('auroc', 0):.3f}          {ar.get('auroc', 0):.3f}")
    print(f"  ECE:              {dlm.get('ece', 0):.3f}          {ar.get('ece', 0):.3f}")
    print(f"  API error rate:   {dlm.get('api_error_rate', 0):.2%}         {ar.get('api_error_rate', 0):.2%}")
    
    comp = analysis.get("comparison", {})
    print(f"\n  DLM vs AR AUROC delta: {comp.get('dlm_vs_ar_auroc_delta', 0):+.3f}")
    print(f"  σ² Spearman ρ(DLM,AR): {comp.get('spearman_rho_sigma2_correlation', 0):.3f}")
    
    gate = analysis.get("decision_gate", {})
    print(f"\n  ┌─ DECISION: {gate.get('decision', '?')} ─────────────────")
    print(f"  │ {gate.get('action', '')}")
    print(f"  └──────────────────────────────────────────────")


# ─── Alternative: Direct HF Inference API via serverless endpoints ────────────

def check_hf_serverless_dlm_availability():
    """
    Check which (if any) DLMs are available on HF Serverless Inference API.
    As of Feb 2026: LLaDA-8B has NO inference providers.
    MDLM and SEDD are smaller but may not have chat fine-tunes available.
    
    This function probes the HF API to find available text-gen models
    that implement masked diffusion.
    """
    import urllib.request
    import urllib.error
    
    # Models to check (public, MIT license, diffusion LMs)
    candidates = [
        "GSAI-ML/LLaDA-8B-Instruct",    # 8B, no providers confirmed Feb 2026
        "GSAI-ML/LLaDA-8B-Base",
        "inclusionAI/LLaDA2.0-mini",     # 16B MoE, 1.4B active — check
        "inclusionAI/LLaDA2.1-mini",     # newer variant
        "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",  # MoE, small active params
    ]
    
    hf_token = os.environ.get("HF_TOKEN", "")
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    results = {}
    for model_id in candidates:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        req = urllib.request.Request(url, headers={**headers, "Content-Type": "application/json"})
        # POST a simple test payload
        payload = json.dumps({"inputs": "Hello, what is 2+2?"}).encode()
        req.data = payload
        
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
                body = resp.read().decode()[:200]
                results[model_id] = {"status": status, "response": body, "available": True}
        except urllib.error.HTTPError as e:
            # 503 = model loading, 404 = not found, 401 = auth required
            results[model_id] = {"status": e.code, "available": False, "error": str(e)}
        except Exception as e:
            results[model_id] = {"status": "error", "available": False, "error": str(e)}
        
        print(f"  {model_id}: {results[model_id]}")
        time.sleep(0.5)
    
    return results


# ─── Setup / Dependency Check ─────────────────────────────────────────────────

def check_dependencies():
    """Check which optional dependencies are available."""
    deps = {}
    
    for pkg in ["gradio_client", "openai", "datasets", "numpy", "scipy", "sentence_transformers"]:
        try:
            __import__(pkg.replace("-", "_"))
            deps[pkg] = True
        except ImportError:
            deps[pkg] = False
    
    print("Dependencies:")
    for pkg, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {pkg}")
    
    return deps


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BPFC Pilot Experiment")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Run with synthetic data (no API calls)")
    parser.add_argument("--n", type=int, default=50, help="Number of questions")
    parser.add_argument("--k", type=int, default=8, help="Passes per question")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies")
    parser.add_argument("--check-api", action="store_true", help="Check HF Inference API availability")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from partial results")
    args = parser.parse_args()
    
    if args.check_deps:
        check_dependencies()
    
    if args.check_api:
        print("\nChecking HF Serverless Inference API for DLMs...")
        results = check_hf_serverless_dlm_availability()
        print(json.dumps(results, indent=2))
    
    if not args.check_deps and not args.check_api:
        # Run the experiment
        analysis = run_pilot(
            n_questions=args.n,
            k_passes=args.k,
            dry_run=args.dry_run,
            resume=not args.no_resume,
        )
        
        print(f"\nResults saved to: {CONFIG['OUTPUT_DIR']}")
