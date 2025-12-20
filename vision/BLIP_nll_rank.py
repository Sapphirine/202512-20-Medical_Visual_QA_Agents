#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============ Environment stabilization (keep at the very top) ============
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# If you need to force synchronous CUDA error pinpointing: uncomment (will be much slower)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import json, math, random, re, inspect, warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image, ImageFile

from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    get_linear_schedule_with_warmup,
)

# Allow loading truncated images (occasionally present in PathVQA/some HF datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", message="Truncated File Read")

# =========================================================
# 0) CONFIG (edit only here)
# =========================================================

# Data: HuggingFace PathVQA
USE_HF_DATASET = True
HF_DATASET_NAME = "flaviagiammarino/path-vqa"
HF_TRAIN_SPLIT = "train"
HF_VAL_SPLIT = "validation"
HF_TEST_SPLIT = "test"
HF_IMAGE_COL = "image"
HF_QUESTION_COL = "question"
HF_ANSWER_COL = "answer"

# If using local JSONL (optional)
DATA_DIR = None
IMAGE_ROOT = None
TRAIN_JSONL = "train.jsonl"
VAL_JSONL = "val.jsonl"
TEST_JSONL = "test.jsonl"

# Model
MODEL_NAME = "Salesforce/blip-vqa-base"

# Output
OUTPUT_DIR = "./runs/blip_pathvqa_best"
BEST_SUBDIR = "best"

# Training hyperparameters
SEED = 42
EPOCHS = 10
BATCH_SIZE = 4
GRAD_ACCUM = 2
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0

# Text lengths
MAX_QUESTION_LEN = 32
MAX_ANSWER_LEN = 8   # teacher-forcing label length (training)
MAX_GEN_LEN = 8      # generation length (evaluation); recommended to match MAX_ANSWER_LEN

USE_PROMPT = True

# DataLoader
NUM_WORKERS = 0         # most stable in shared/Notebook env; increase to 2/4 once stable
PIN_MEMORY = True

# Mixed precision
USE_FP16 = True
USE_BF16 = False

# Generation parameters for evaluation (aligned with your standalone evaluation)
NUM_BEAMS = 3

# ====== NLL ranking inference (answer selection) ======
# Evaluation method: "generate" = original generate()+beam; "nll" = candidate answer NLL ranking (more like classification)
EVAL_METHOD = "nll"   # recommended for PathVQA
# NLL candidates: take top-K most frequent answers from training set; -1 means all unique answers from training set
NLL_CANDIDATE_TOPK = 1000
# For NLL scoring, how many candidates to compute in parallel each time (larger is faster but uses more VRAM)
NLL_CHUNK_SIZE = 64
# Deduplicate candidate answers by normalized text (recommended True to reduce duplicates)
NLL_DEDUP_NORMALIZED = True

# Whether to print some val samples during training (style aligned with standalone evaluation)
EVAL_PRINT_SAMPLES = 10

# Which metric determines the best checkpoint (favor accuracy-first)
# Options: "exact_acc", "token_f1", "open_exact_acc", "yesno_acc"
BEST_METRIC = "exact_acc"

# Whether to enable constrained decoding during evaluation (default off: aligned with standalone evaluation)
EVAL_CONSTRAINED_DECODE = False
ANSWER_VOCAB_SIZE = -1   # for constrained decoding; -1 = all training answers

# Debug/preflight
CPU_PREFLIGHT_CHECK = True
DEBUG_SANITY_STEPS = 3   # assert token range for the first few global steps

# =========================================================
# 1) Metrics (fully aligned with your standalone evaluation script)
# =========================================================

def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def token_f1(pred: str, gt: str) -> float:
    p = pred.split()
    g = gt.split()
    if len(p) == 0 or len(g) == 0:
        return 0.0
    common = set(p) & set(g)
    if len(common) == 0:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)

# =========================================================
# 2) utils
# =========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# =========================================================
# 3) dataset
# =========================================================

class HFVQADataset(Dataset):
    """
    Note: returns (image, question, answer_raw)
    The answer is not normalized (training needs raw answer tokenization).
    """
    def __init__(self, name: str, split: str, image_col: str, q_col: str, a_col: str):
        from datasets import load_dataset
        self.ds = load_dataset(name, split=split)
        self.image_col = image_col
        self.q_col = q_col
        self.a_col = a_col

        for c in [image_col, q_col, a_col]:
            if c not in self.ds.column_names:
                raise ValueError(f"Column '{c}' not in {self.ds.column_names}")

        # Cache raw answers so building the trie won't load all images
        self._answers_raw = [str(x) for x in list(self.ds[a_col])]

    def get_answers_raw(self) -> List[str]:
        return self._answers_raw

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str, str]:
        ex = self.ds[idx]
        img = ex[self.image_col]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img).convert("RGB")
        q = str(ex[self.q_col])
        a = self._answers_raw[idx]
        return img.convert("RGB"), q, a


class JsonlVQADataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: Optional[str]):
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
        self.samples: List[Dict[str, Any]] = []
        self.image_root = image_root
        self._answers_raw: List[str] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                for k in ["image", "question", "answer"]:
                    if k not in ex:
                        raise ValueError(f"{jsonl_path} line {i} missing key '{k}', got keys={list(ex.keys())}")
                self.samples.append(ex)
                self._answers_raw.append(str(ex["answer"]))

    def get_answers_raw(self) -> List[str]:
        return self._answers_raw

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str, str]:
        ex = self.samples[idx]
        img_path = ex["image"]
        if self.image_root and not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        q = str(ex["question"])
        a = str(ex["answer"])
        return img, q, a

# =========================================================
# 4) constrained decoding (optional)
# =========================================================

class TrieNode:
    __slots__ = ("children", "is_end")
    def __init__(self):
        self.children: Dict[int, "TrieNode"] = {}
        self.is_end: bool = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_ids: List[int]) -> None:
        node = self.root
        for t in token_ids:
            if t not in node.children:
                node.children[t] = TrieNode()
            node = node.children[t]
        node.is_end = True

    def next_tokens(self, prefix: List[int]) -> Tuple[List[int], bool]:
        node = self.root
        for t in prefix:
            if t not in node.children:
                return [], False
            node = node.children[t]
        return list(node.children.keys()), node.is_end

def build_answer_set_raw(ds: Dataset, max_answers: int) -> List[str]:
    if hasattr(ds, "get_answers_raw"):
        answers = ds.get_answers_raw()
    else:
        answers = [ds[i][2] for i in range(len(ds))]
    cnt = Counter(answers)
    if max_answers is not None and max_answers > 0:
        ans_list = [a for a, _ in cnt.most_common(max_answers)]
    else:
        ans_list = list(cnt.keys())
    # Force include yes/no (case might differ in raw, that's fine)
    for a in ["yes", "no", "Yes", "No"]:
        if a not in ans_list:
            ans_list.append(a)
    return ans_list

def build_trie(answer_list: List[str], tokenizer) -> Trie:
    trie = Trie()
    for ans in answer_list:
        ids = tokenizer(ans, add_special_tokens=False).input_ids
        if ids:
            trie.insert(ids)
    return trie

def make_prefix_allowed_tokens_fn(trie: Trie, bos_id: Optional[int], eos_id: int):
    def _fn(batch_id: int, input_ids: torch.LongTensor) -> List[int]:
        ids = input_ids.tolist()
        if bos_id is not None and len(ids) > 0 and ids[0] == bos_id:
            prefix = ids[1:]
        else:
            prefix = ids
        nxt, is_end = trie.next_tokens(prefix)
        allowed = list(nxt)
        if is_end:
            allowed.append(eos_id)
        if not allowed:
            allowed = [eos_id]
        return allowed
    return _fn

# =========================================================
# 5) collate (process image/text separately to ensure truncation takes effect)
# =========================================================

@dataclass
class Batch:
    pixel_values: torch.FloatTensor
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    labels: torch.LongTensor
    answers_raw: List[str]
    questions_raw: List[str]

def make_collate_fn(processor: BlipProcessor, max_q_len: int, max_a_len: int):
    tok = processor.tokenizer

    def _collate(examples: List[Tuple[Image.Image, str, str]]) -> Batch:
        images, questions, answers = zip(*examples)

        questions_in = list(questions)
        if USE_PROMPT:
            questions_in = [f"Question: {q} Answer:" for q in questions_in]

        # Images
        pixel = processor(images=list(images), return_tensors="pt")

        # Questions (force tokenizer truncation)
        txt = tok(
            questions_in,
            padding=True,
            truncation=True,
            max_length=max_q_len,
            return_tensors="pt",
        )

        # Answer labels (teacher-forcing)
        ans_tok = tok(
            list(answers),
            padding=True,
            truncation=True,
            max_length=max_a_len,
            return_tensors="pt",
            add_special_tokens=True,
        )
        labels = ans_tok.input_ids.clone()
        labels[labels == tok.pad_token_id] = -100

        return Batch(
            pixel_values=pixel["pixel_values"],
            input_ids=txt["input_ids"],
            attention_mask=txt["attention_mask"],
            labels=labels,
            answers_raw=list(answers),
            questions_raw=list(questions),
        )

    return _collate

# =========================================================
# 6) decode + evaluate (aligned with standalone evaluation)
# =========================================================

def decode_one(gen_ids_1d: List[int], tokenizer, bos_id, eos_id, pad_id) -> str:
    ids = list(gen_ids_1d)
    if bos_id is not None and len(ids) > 0 and ids[0] == bos_id:
        ids = ids[1:]
    if eos_id is not None and eos_id in ids:
        ids = ids[:ids.index(eos_id)]
    if pad_id is not None:
        ids = [t for t in ids if t != pad_id]
    return tokenizer.decode(ids, skip_special_tokens=True)

@torch.no_grad()
def evaluate_generation(
    model,
    processor,
    loader,
    device,
    use_amp,
    amp_dtype,
    max_gen_len: int,
    num_beams: int,
    prefix_allowed_tokens_fn=None,
    print_samples: int = 0,
):
    tok = processor.tokenizer
    pad_id = tok.pad_token_id
    bos_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    eos_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id

    total = 0
    exact = 0
    f1_sum = 0.0

    yesno_total = 0
    yesno_correct = 0

    open_total = 0
    open_exact = 0
    open_f1_sum = 0.0

    printed = 0

    model.eval()
    for batch in tqdm(loader, desc="eval(gen)", leave=False):
        pixel_values = batch.pixel_values.to(device, non_blocking=True)
        input_ids = batch.input_ids.to(device, non_blocking=True)
        attention_mask = batch.attention_mask.to(device, non_blocking=True)

        gen_kwargs = dict(
            max_length=1 + max_gen_len,
            num_beams=num_beams,
            do_sample=False,
        )
        if prefix_allowed_tokens_fn is not None:
            gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            gen_out = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        # per-sample
        for seq, gt_raw, q_raw in zip(gen_out, batch.answers_raw, batch.questions_raw):
            pred_raw = decode_one(seq.tolist(), tok, bos_id, eos_id, pad_id)

            pred = normalize_answer(pred_raw)
            gt = normalize_answer(gt_raw)

            total += 1
            if pred == gt:
                exact += 1
            f1_sum += token_f1(pred, gt)

            if gt in ["yes", "no"]:
                yesno_total += 1
                if pred == gt:
                    yesno_correct += 1
            else:
                open_total += 1
                if pred == gt:
                    open_exact += 1
                open_f1_sum += token_f1(pred, gt)

            if printed < print_samples:
                print("=" * 60)
                print("Q :", q_raw)
                print("GT:", gt)
                print("PR:", pred)
                printed += 1

    metrics = {
        "exact_acc": exact / max(1, total),
        "token_f1": f1_sum / max(1, total),
        "yesno_acc": yesno_correct / max(1, yesno_total),
        "open_exact_acc": open_exact / max(1, open_total),
        "open_token_f1": open_f1_sum / max(1, open_total),
        "n": total,
        "n_yesno": yesno_total,
        "n_open": open_total,
    }
    return metrics


# =========================================================
# 6.5) NLL ranking (score candidate answers and choose the best; replaces generate)
# =========================================================

def dedup_answer_list_by_normalized(answer_list: List[str]) -> List[str]:
    """
    Deduplicate candidate answers by normalize_answer, keeping the first occurrence's raw form.
    If build_answer_set_raw() returns answers sorted by frequency, the first occurrence tends to be the higher-frequency wording.
    """
    seen = set()
    deduped = []
    for a in answer_list:
        n = normalize_answer(a)
        if n and n not in seen:
            seen.add(n)
            deduped.append(a)
    return deduped

def prepare_candidate_tensors(
    answer_list_raw: List[str],
    tokenizer,
    max_a_len: int,
    bos_id: int,
    pad_id: int,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    Precompute tensors for NLL ranking candidates:
      - label_ids: (M, T) target tokens (with special tokens), padded with pad_id
      - dec_ids:   (M, T) decoder input (labels shifted right by one; BOS at position 0)
      - loss_mask: (M, T) which positions contribute to NLL (non-pad; exclude position 0)
    """
    tok_out = tokenizer(
        answer_list_raw,
        padding="max_length",
        truncation=True,
        max_length=max_a_len,
        return_tensors="pt",
        add_special_tokens=True,
    )
    label_ids = tok_out.input_ids  # (M, T), pad = pad_id

    # shift-right: dec[t] as input to predict label[t]
    dec_ids = label_ids.clone()
    dec_ids = torch.roll(dec_ids, shifts=1, dims=1)
    dec_ids[:, 0] = bos_id

    loss_mask = (label_ids != pad_id).long()
    # do not count position 0 (usually CLS/BOS); start NLL from position 1
    if loss_mask.size(1) > 0:
        loss_mask[:, 0] = 0

    return label_ids, dec_ids, loss_mask

def _nll_from_logits(
    logits: torch.FloatTensor,
    target_ids: torch.LongTensor,
    loss_mask: torch.LongTensor,
) -> torch.FloatTensor:
    """
    logits:    (N, T, V)
    target:    (N, T)
    loss_mask: (N, T) 0/1
    return:    (N,) token-level NLL sum per sequence (sum after masking)

    Notes:
      - Avoid explicitly building (N,T,V) log_probs (very memory-heavy)
      - Using cross_entropy(reduction='none') to get per-token loss saves memory
    """
    N, T, V = logits.shape
    # (N*T, V) vs (N*T,)
    per_token = torch.nn.functional.cross_entropy(
        logits.reshape(N * T, V),
        target_ids.reshape(N * T),
        reduction="none",
    ).reshape(N, T)  # (N, T)
    nll = (per_token * loss_mask.float()).sum(dim=1)  # (N,)
    return nll
@torch.no_grad()
def evaluate_nll_ranking(
    model,
    processor,
    loader,
    device,
    use_amp: bool,
    amp_dtype,
    cand_answers_raw: List[str],
    cand_answers_norm: List[str],
    cand_label_ids: torch.LongTensor,
    cand_dec_ids: torch.LongTensor,
    cand_loss_mask: torch.LongTensor,
    chunk_size: int,
    print_samples: int = 0,
):
    """
    Use a candidate answer set for NLL ranking:
      pred = argmin_a NLL(a | image, question)
    Returns the same metrics as evaluate_generation: exact_acc/token_f1/yesno_acc/open_*
    """
    tok = processor.tokenizer
    pad_id = tok.pad_token_id

    total = 0
    exact = 0
    f1_sum = 0.0

    yesno_total = 0
    yesno_correct = 0

    open_total = 0
    open_exact = 0
    open_f1_sum = 0.0

    printed = 0

    # Put candidate tensors on device (do once)
    cand_label_ids = cand_label_ids.to(device)
    cand_dec_ids = cand_dec_ids.to(device)
    cand_loss_mask = cand_loss_mask.to(device)

    # Try a faster path: compute image->question_embeds once, then run only the decoder
    fast_path = all(hasattr(model, k) for k in ["vision_model", "text_encoder", "text_decoder"])

    model.eval()
    for batch in tqdm(loader, desc="eval(nll-rank)", leave=False):
        pixel_values = batch.pixel_values.to(device, non_blocking=True)
        input_ids = batch.input_ids.to(device, non_blocking=True)
        attention_mask = batch.attention_mask.to(device, non_blocking=True)

        B = input_ids.size(0)
        M, T = cand_label_ids.size(0), cand_label_ids.size(1)

        # --------- 1) Encode: image + question (once per batch) ---------
        question_embeds = None
        if fast_path:
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                image_outputs = model.vision_model(pixel_values=pixel_values, return_dict=True)
                image_embeds = image_outputs.last_hidden_state
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

                q_outputs = model.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                question_embeds = q_outputs.last_hidden_state  # (B, Lq, H)

        # --------- 2) Score all candidate answers via NLL (chunked) ---------
        best_nll = torch.full((B,), float("inf"), device=device)
        best_idx = torch.zeros((B,), dtype=torch.long, device=device)

        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            C = end - start

            c_label = cand_label_ids[start:end]      # (C, T)
            c_dec = cand_dec_ids[start:end]          # (C, T)
            c_mask = cand_loss_mask[start:end]       # (C, T)

            # (B, C, T) -> (B*C, T)
            label_rep = c_label.unsqueeze(0).expand(B, C, T).reshape(B * C, T)
            dec_rep = c_dec.unsqueeze(0).expand(B, C, T).reshape(B * C, T)
            mask_rep = c_mask.unsqueeze(0).expand(B, C, T).reshape(B * C, T)

            dec_attn_rep = (dec_rep != pad_id).long()

            if question_embeds is not None:
                # Decoder-only path: encoder_hidden_states = question_embeds
                Lq = question_embeds.size(1)
                H = question_embeds.size(2)

                q_rep = question_embeds.unsqueeze(1).expand(B, C, Lq, H).reshape(B * C, Lq, H)
                enc_attn_rep = attention_mask.unsqueeze(1).expand(B, C, Lq).reshape(B * C, Lq)

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    dec_out = model.text_decoder(
                        input_ids=dec_rep,
                        attention_mask=dec_attn_rep,
                        encoder_hidden_states=q_rep,
                        encoder_attention_mask=enc_attn_rep,
                        return_dict=True,
                    )
                    logits = dec_out.logits  # (B*C, T, V)
            else:
                # Fallback: run the full forward pass (slower, but more compatible)
                pixel_rep = pixel_values.unsqueeze(1).expand(B, C, *pixel_values.shape[1:]).reshape(B * C, *pixel_values.shape[1:])
                input_rep = input_ids.unsqueeze(1).expand(B, C, input_ids.size(1)).reshape(B * C, input_ids.size(1))
                attn_rep = attention_mask.unsqueeze(1).expand(B, C, attention_mask.size(1)).reshape(B * C, attention_mask.size(1))

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    out = model(
                        pixel_values=pixel_rep,
                        input_ids=input_rep,
                        attention_mask=attn_rep,
                        decoder_input_ids=dec_rep,
                        decoder_attention_mask=dec_attn_rep,
                        return_dict=True,
                    )
                    logits = out.logits

            nll_pair = _nll_from_logits(logits, label_rep, mask_rep)  # (B*C,)
            nll_pair = nll_pair.view(B, C)                            # (B, C)

            chunk_best_nll, chunk_best_pos = nll_pair.min(dim=1)      # (B,)
            update = chunk_best_nll < best_nll
            best_nll = torch.where(update, chunk_best_nll, best_nll)
            best_idx = torch.where(update, (chunk_best_pos + start).to(best_idx.dtype), best_idx)

        # --------- 3) Metrics (aligned with standalone evaluation) ---------
        best_idx_cpu = best_idx.detach().cpu().tolist()
        for idx, gt_raw, q_raw in zip(best_idx_cpu, batch.answers_raw, batch.questions_raw):
            pred = cand_answers_norm[idx]
            gt = normalize_answer(gt_raw)

            total += 1
            if pred == gt:
                exact += 1
            f1_sum += token_f1(pred, gt)

            if gt in ["yes", "no"]:
                yesno_total += 1
                if pred == gt:
                    yesno_correct += 1
            else:
                open_total += 1
                if pred == gt:
                    open_exact += 1
                open_f1_sum += token_f1(pred, gt)

            if printed < print_samples:
                print("=" * 60)
                print("Q :", q_raw)
                print("GT:", gt)
                print("PR:", pred)
                printed += 1

    metrics = {
        "exact_acc": exact / max(1, total),
        "token_f1": f1_sum / max(1, total),
        "yesno_acc": yesno_correct / max(1, yesno_total),
        "open_exact_acc": open_exact / max(1, open_total),
        "open_token_f1": open_f1_sum / max(1, open_total),
        "n": total,
        "n_yesno": yesno_total,
        "n_open": open_total,
    }
    return metrics

# =========================================================
# 7) main
# =========================================================

def main():
    if USE_FP16 and USE_BF16:
        raise ValueError("Choose only one: USE_FP16 or USE_BF16")

    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_FP16 or USE_BF16
    amp_dtype = torch.float16 if USE_FP16 else (torch.bfloat16 if USE_BF16 else torch.float32)
    print(f"Device={device}  AMP={use_amp}  dtype={amp_dtype}")

    # Load on CPU first to fix pad/config + preflight
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME)
    tok = processor.tokenizer

    # Ensure pad_token exists (avoid label shift pitfalls)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tok))

    pad_id = tok.pad_token_id
    bos_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    eos_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    if bos_id is None or eos_id is None:
        raise ValueError("Cannot determine BOS/EOS token id for BLIP tokenizer.")

    # Sync config (critical)
    model.config.pad_token_id = pad_id
    model.config.bos_token_id = bos_id
    model.config.eos_token_id = eos_id
    model.config.decoder_start_token_id = bos_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = pad_id
        model.generation_config.bos_token_id = bos_id
        model.generation_config.eos_token_id = eos_id

    # Max lengths (avoid position embedding overflow)
    max_pos = int(model.text_encoder.config.max_position_embeddings)
    max_q_len = min(MAX_QUESTION_LEN, max_pos)
    max_a_len = min(MAX_ANSWER_LEN, max_pos)
    print(f"[Len] max_pos={max_pos}, use max_q_len={max_q_len}, max_a_len={max_a_len}, max_gen_len={MAX_GEN_LEN}")
    print(f"[Tokens] pad={pad_id} bos={bos_id} eos={eos_id} vocab(len(tok))={len(tok)}")

    # Load dataset
    if USE_HF_DATASET:
        train_ds = HFVQADataset(HF_DATASET_NAME, HF_TRAIN_SPLIT, HF_IMAGE_COL, HF_QUESTION_COL, HF_ANSWER_COL)
        val_ds = HFVQADataset(HF_DATASET_NAME, HF_VAL_SPLIT, HF_IMAGE_COL, HF_QUESTION_COL, HF_ANSWER_COL)
        test_ds = HFVQADataset(HF_DATASET_NAME, HF_TEST_SPLIT, HF_IMAGE_COL, HF_QUESTION_COL, HF_ANSWER_COL)
    else:
        if DATA_DIR is None or not os.path.isdir(DATA_DIR):
            raise ValueError("Set USE_HF_DATASET=True or provide valid DATA_DIR for local JSONL.")
        image_root = IMAGE_ROOT if IMAGE_ROOT is not None else DATA_DIR
        train_ds = JsonlVQADataset(os.path.join(DATA_DIR, TRAIN_JSONL), image_root=image_root)
        val_ds = JsonlVQADataset(os.path.join(DATA_DIR, VAL_JSONL), image_root=image_root)
        test_ds = JsonlVQADataset(os.path.join(DATA_DIR, TEST_JSONL), image_root=image_root)

    print(f"Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")

    # ====== Candidate answers for NLL ranking (built from TRAIN answers) ======
    cand_answers_raw = None
    cand_answers_norm = None
    cand_label_ids = None
    cand_dec_ids = None
    cand_loss_mask = None
    if EVAL_METHOD.lower() == "nll":
        cand_answers_raw = build_answer_set_raw(train_ds, NLL_CANDIDATE_TOPK)
        if NLL_DEDUP_NORMALIZED:
            cand_answers_raw = dedup_answer_list_by_normalized(cand_answers_raw)
        # Force include yes/no (avoid missing them in the candidate set)
        for a in ["yes", "no", "Yes", "No"]:
            if a not in cand_answers_raw:
                cand_answers_raw.append(a)

        cand_answers_norm = [normalize_answer(a) for a in cand_answers_raw]
        cand_label_ids, cand_dec_ids, cand_loss_mask = prepare_candidate_tensors(
            cand_answers_raw, tok, max_a_len=max_a_len, bos_id=bos_id, pad_id=pad_id
        )
        print(
            f"[NLL Ranking] candidates={len(cand_answers_raw)} "
            f"topK={NLL_CANDIDATE_TOPK} chunk={NLL_CHUNK_SIZE} dedup={NLL_DEDUP_NORMALIZED}"
        )

    collate_fn = make_collate_fn(processor, max_q_len=max_q_len, max_a_len=max_a_len)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )

    # ---------- CPU preflight: catch index out-of-range ----------
    if CPU_PREFLIGHT_CHECK:
        print("[Preflight] Running 1 batch on CPU...")
        model.eval()
        batch = next(iter(train_loader))
        with torch.no_grad():
            V = len(tok)
            assert int(batch.input_ids.min()) >= 0
            assert int(batch.input_ids.max()) < V
            mask = batch.labels != -100
            if mask.any():
                assert int(batch.labels[mask].min()) >= 0
                assert int(batch.labels[mask].max()) < V

            fwd_params = inspect.signature(model.forward).parameters
            supports_decoder = "decoder_input_ids" in fwd_params

            kwargs = dict(
                pixel_values=batch.pixel_values,
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
            )
            if supports_decoder:
                dec = batch.labels.clone()
                dec[dec == -100] = pad_id
                dec = torch.roll(dec, shifts=1, dims=1)
                dec[:, 0] = bos_id
                kwargs["decoder_input_ids"] = dec
                if "decoder_attention_mask" in fwd_params:
                    kwargs["decoder_attention_mask"] = (dec != pad_id).long()

            _ = model(**kwargs)

        print("[Preflight] OK.")

    # Evaluation: optional constrained decoding (default off to match standalone eval)
    prefix_allowed_tokens_fn = None
    if EVAL_CONSTRAINED_DECODE:
        ans_list = build_answer_set_raw(train_ds, ANSWER_VOCAB_SIZE)
        trie = build_trie(ans_list, tok)
        prefix_allowed_tokens_fn = make_prefix_allowed_tokens_fn(trie, bos_id=bos_id, eos_id=eos_id)
        print(f"[Eval ConstrainedDecode] answer_set={len(ans_list)}")

    # Move to GPU
    model.to(device)
    model.config.use_cache = False

    # optimizer / scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = math.ceil(len(train_loader) / max(1, GRAD_ACCUM)) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)

    # Whether forward supports decoder_input_ids
    fwd_params = inspect.signature(model.forward).parameters
    supports_decoder = "decoder_input_ids" in fwd_params
    supports_decoder_attn = "decoder_attention_mask" in fwd_params

    # Save config
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        cfg = {k: v for k, v in globals().items() if k.isupper()}
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    best_score = -1e9
    best_metrics = None
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            pixel_values = batch.pixel_values.to(device, non_blocking=True)
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)

            # Sanity checks (only for the first few global steps)
            if DEBUG_SANITY_STEPS > 0 and global_step < DEBUG_SANITY_STEPS:
                V = len(tok)
                assert int(input_ids.min()) >= 0 and int(input_ids.max()) < V

            kwargs = dict(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Manually construct decoder_input_ids (avoid internal shift pitfalls)
            if supports_decoder:
                dec = labels.clone()
                dec[dec == -100] = pad_id
                dec = torch.roll(dec, shifts=1, dims=1)
                dec[:, 0] = bos_id
                kwargs["decoder_input_ids"] = dec
                if supports_decoder_attn:
                    kwargs["decoder_attention_mask"] = (dec != pad_id).long()

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                out = model(**kwargs)
                loss = out.loss / max(1, GRAD_ACCUM)

            if USE_FP16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * max(1, GRAD_ACCUM)

            if step % GRAD_ACCUM == 0:
                if USE_FP16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 50 == 0:
                    pbar.set_postfix(loss=f"{running_loss / max(1, global_step):.4f}")

        # ====== VAL EVAL (aligned with standalone evaluation logic) ======
        if EVAL_METHOD.lower() == "nll":
            val_metrics = evaluate_nll_ranking(
                model, processor, val_loader, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
                cand_answers_raw=cand_answers_raw,
                cand_answers_norm=cand_answers_norm,
                cand_label_ids=cand_label_ids,
                cand_dec_ids=cand_dec_ids,
                cand_loss_mask=cand_loss_mask,
                chunk_size=NLL_CHUNK_SIZE,
                print_samples=EVAL_PRINT_SAMPLES if epoch == 1 else 0,
            )
        else:
            val_metrics = evaluate_generation(
                model, processor, val_loader, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
                max_gen_len=MAX_GEN_LEN, num_beams=NUM_BEAMS,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                print_samples=EVAL_PRINT_SAMPLES if epoch == 1 else 0,
            )

        print(
            f"[Epoch {epoch}] VAL "
            f"exact={val_metrics['exact_acc']:.4f} "
            f"f1={val_metrics['token_f1']:.4f} "
            f"yesno={val_metrics['yesno_acc']:.4f} "
            f"open_exact={val_metrics['open_exact_acc']:.4f} "
            f"open_f1={val_metrics['open_token_f1']:.4f} "
            f"(n={val_metrics['n']}, yesno={val_metrics['n_yesno']}, open={val_metrics['n_open']})"
        )

        # Save best
        score = float(val_metrics.get(BEST_METRIC, -1e9))
        if score > best_score:
            best_score = score
            best_metrics = val_metrics

            best_dir = os.path.join(OUTPUT_DIR, BEST_SUBDIR)
            ensure_dir(best_dir)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)

            with open(os.path.join(best_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(val_metrics, f, ensure_ascii=False, indent=2)

            print(f"  -> saved BEST by {BEST_METRIC}={best_score:.4f} to {best_dir}")

    # ====== TEST BEST ======
    print("\nLoading BEST checkpoint for TEST...")
    best_dir = os.path.join(OUTPUT_DIR, BEST_SUBDIR)
    model = BlipForQuestionAnswering.from_pretrained(best_dir).to(device)
    processor = BlipProcessor.from_pretrained(best_dir)

    if EVAL_METHOD.lower() == "nll":
        # Tokenizer may change after save/load; rebuild candidate tensors with the current processor.tokenizer
        tok = processor.tokenizer
        pad_id = tok.pad_token_id
        bos_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
        if bos_id is None or pad_id is None:
            raise ValueError("Tokenizer must have BOS/PAD token for NLL ranking.")

        cand_answers_norm = [normalize_answer(a) for a in cand_answers_raw]
        cand_label_ids, cand_dec_ids, cand_loss_mask = prepare_candidate_tensors(
            cand_answers_raw, tok, max_a_len=max_a_len, bos_id=bos_id, pad_id=pad_id
        )

        test_metrics = evaluate_nll_ranking(
            model, processor, test_loader, device,
            use_amp=use_amp, amp_dtype=amp_dtype,
            cand_answers_raw=cand_answers_raw,
            cand_answers_norm=cand_answers_norm,
            cand_label_ids=cand_label_ids,
            cand_dec_ids=cand_dec_ids,
            cand_loss_mask=cand_loss_mask,
            chunk_size=NLL_CHUNK_SIZE,
            print_samples=EVAL_PRINT_SAMPLES,
        )
    else:
        test_metrics = evaluate_generation(
            model, processor, test_loader, device,
            use_amp=use_amp, amp_dtype=amp_dtype,
            max_gen_len=MAX_GEN_LEN, num_beams=NUM_BEAMS,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            print_samples=EVAL_PRINT_SAMPLES,
        )

    print(
        f"[BEST] TEST "
        f"exact={test_metrics['exact_acc']:.4f} "
        f"f1={test_metrics['token_f1']:.4f} "
        f"yesno={test_metrics['yesno_acc']:.4f} "
        f"open_exact={test_metrics['open_exact_acc']:.4f} "
        f"open_f1={test_metrics['open_token_f1']:.4f} "
        f"(n={test_metrics['n']}, yesno={test_metrics['n_yesno']}, open={test_metrics['n_open']})"
    )

    with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()