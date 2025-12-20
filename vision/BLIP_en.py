#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============ Environment Stabilization (keep early) ============
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# If you need synchronous CUDA debugging, uncomment (will slow down significantly)
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

# Allow loading truncated images (PathVQA/some HF datasets may have them)
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", message="Truncated File Read")

# =========================================================
# 0) CONFIG (only edit here)
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

# Text length
MAX_QUESTION_LEN = 32
MAX_ANSWER_LEN = 8   # teacher forcing label length (training)
MAX_GEN_LEN = 8      # generation length for inference (evaluation); recommend same as MAX_ANSWER_LEN

USE_PROMPT = True

# DataLoader
NUM_WORKERS = 0         # most stable for shared/Notebook env; increase to 2/4 after it's stable
PIN_MEMORY = True

# Mixed precision
USE_FP16 = True
USE_BF16 = False

# Generation params for eval (aligned with standalone evaluation)
NUM_BEAMS = 3

# Whether to print some val samples during training (aligned with standalone evaluation logging)
EVAL_PRINT_SAMPLES = 10

# Metric to decide best checkpoint (prioritize accuracy)
# Options: "exact_acc", "token_f1", "open_exact_acc", "yesno_acc"
BEST_METRIC = "exact_acc"

# Enable constrained decoding during eval? (off by default to match standalone evaluation)
EVAL_CONSTRAINED_DECODE = False
ANSWER_VOCAB_SIZE = -1   # for constrained decode; -1 = all training answers

# Debug/preflight
CPU_PREFLIGHT_CHECK = True
DEBUG_SANITY_STEPS = 3   # assert token range for the first few steps

# =========================================================
# 1) Metrics (fully aligned with standalone evaluation script)
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
    Note: returns (image, question, answer_raw).
    The answer is NOT normalized (training uses raw answer tokenization).
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

        # Cache raw answers so building trie doesn't load all images
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
    # Force include yes/no (case may vary in raw but it's fine)
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
# 5) collate (separate image/text to ensure truncation is applied)
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

        # Answer labels (teacher forcing)
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
# 6) decode + evaluate (match standalone evaluation)
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

    # # Load on CPU first to fix pad/config + preflight
    # processor = BlipProcessor.from_pretrained(MODEL_NAME)
    # model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME)
    # tok = processor.tokenizer

    best_dir = os.path.join(OUTPUT_DIR, BEST_SUBDIR)
    best_dir = "./runs/blip_pathvqa_best/best"

    if os.path.isdir(best_dir):
        print(f"[Resume-best] Loading model from {best_dir}")
        processor = BlipProcessor.from_pretrained(best_dir)
        model = BlipForQuestionAnswering.from_pretrained(best_dir)
    else:
        print(f"[Resume-best] best/ not found, loading base model {MODEL_NAME}")
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

    # Sync config (important)
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

    # CPU preflight: catch index out-of-range
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

    # Eval: optionally constrained decoding (default off to match standalone eval)
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

            # sanity (only check the first few global steps)
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

        # ====== VAL EVAL (aligned with standalone logic) ======
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

