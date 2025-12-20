import os
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import re

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPVisionModel,
    CLIPImageProcessor,
)

# ======================================================
# 1. Device
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ======================================================
# 2. Hyperparameter: number of visual tokens
# ======================================================
NUM_VISION_TOKENS = 4  # Try 2/4/8; 4 is usually much better than 1

# ======================================================
# 3. Load CLIP vision model (instead of ResNet)
# ======================================================
clip_name = "openai/clip-vit-base-patch32"
clip_vision = CLIPVisionModel.from_pretrained(clip_name).to(device)
clip_processor = CLIPImageProcessor.from_pretrained(clip_name)

# Freeze CLIP (train projector first; unfreeze later if needed)
for p in clip_vision.parameters():
    p.requires_grad = False

vision_width = clip_vision.config.hidden_size  # typically 768
print("CLIP vision loaded. hidden_size =", vision_width)

# ======================================================
# 4. Load TinyLLaMA (fp16 on GPU)
# ======================================================
llama_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(llama_name)
llama = AutoModelForCausalLM.from_pretrained(
    llama_name,
    torch_dtype=torch.float16,
).to(device)

# Freeze LLaMA params, train projector only
for p in llama.parameters():
    p.requires_grad = False

llama.train()   # Training mode; params are frozen, grads flow only via inputs_embeds
lm_hidden_size = llama.config.hidden_size      # typically 2048
print("TinyLLaMA loaded. hidden_size =", lm_hidden_size)

# ======================================================
# 5. Projector: projection supporting multiple visual tokens
#    Input (B, C) or (B, N, C); output (B, N, H)
# ======================================================
class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        """
        x: (B, C) or (B, N, C)
        return: (B, N, out_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, C)
        # nn.Linear maps along the last dimension; earlier dims are broadcast
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

projector = Projector(vision_width, lm_hidden_size).to(device)  # on GPU
print("Projector loaded on", device)

# ======================================================
# 6. Evaluation metrics
# ======================================================
def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())

def f1(pred, gt):
    pt = normalize(pred).split()
    gt = normalize(gt).split()
    if len(pt) == 0 or len(gt) == 0:
        return float(pt == gt)
    common = set(pt) & set(gt)
    if not common:
        return 0.0
    p = len(common) / len(pt)
    r = len(common) / len(gt)
    return 2 * p * r / (p + r)

def em(pred, gt):
    return float(normalize(pred) == normalize(gt))

# ======================================================
# 7. Load PathVQA train / test
# ======================================================
train_ds = load_dataset("flaviagiammarino/path-vqa", split="train")
test_ds  = load_dataset("flaviagiammarino/path-vqa", split="test")

# Train/test on full dataset (optionally truncate)
MAX_TRAIN_SAMPLES = None
MAX_TEST_SAMPLES  = None

if MAX_TRAIN_SAMPLES:
    train_ds = train_ds.select(range(MAX_TRAIN_SAMPLES))
if MAX_TEST_SAMPLES:
    test_ds = test_ds.select(range(MAX_TEST_SAMPLES))

# During training use the first 1000 samples for quick validation
VAL_SAMPLES_FOR_EPOCH = 1000
val_ds = test_ds.select(
    range(min(VAL_SAMPLES_FOR_EPOCH, len(test_ds)))
)

print("Train set:", len(train_ds), " Test set:", len(test_ds))
print("Val during training (per epoch):", len(val_ds))

# ======================================================
# 8. Multi-visual-token fusion: build N visual tokens from CLIP patch tokens
# ======================================================
def extract_vision_embeds(img):
    """
    Input: PIL Image
    Output: (1, NUM_VISION_TOKENS, lm_hidden_size) sequence of visual tokens (fp16)
    """
    clip_inputs = clip_processor(
        images=img,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        vision_outputs = clip_vision(**clip_inputs)
        # last_hidden_state: (B, 1 + patch_num, C), index 0 is CLS
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]  # drop CLS, keep all patches

    B, S, C = patch_tokens.shape
    num_tokens = NUM_VISION_TOKENS

    # If patches are fewer than needed, repeat (rare)
    if S < num_tokens:
        repeat_factor = (num_tokens + S - 1) // S
        patch_tokens = patch_tokens.repeat(1, repeat_factor, 1)
        S = patch_tokens.size(1)

    # Split S patches into num_tokens groups and mean-pool each group
    chunk_size = S // num_tokens
    usable_len = num_tokens * chunk_size
    patch_tokens = patch_tokens[:, :usable_len, :]                    # (B, num_tokens*chunk, C)
    patch_tokens = patch_tokens.view(B, num_tokens, chunk_size, C)    # (B, num_tokens, chunk, C)
    patch_tokens = patch_tokens.mean(dim=2)                           # (B, num_tokens, C)

    # Map to LLaMA hidden space via the projector
    vision_embeds_f32 = projector(patch_tokens)            # (B, num_tokens, H), float32
    vision_embeds = vision_embeds_f32.to(torch.float16)    # cast to fp16 for LLaMA

    return vision_embeds  # (1, NUM_VISION_TOKENS, H)

# ======================================================
# 9. Build inputs + labels (account for N visual tokens)
# ======================================================
def build_inputs_and_labels(question, answer, num_vision_tokens):
    prompt = f"Question: {question}\nAnswer:"
    full   = f"{prompt} {answer}"

    tok_full   = tokenizer(full, return_tensors="pt")
    tok_prompt = tokenizer(prompt, return_tensors="pt")

    input_ids_full = tok_full.input_ids.to(device)        # (1, L_full)
    input_ids_prompt = tok_prompt.input_ids.to(device)    # (1, L_prompt)

    L_full   = input_ids_full.size(1)
    L_prompt = input_ids_prompt.size(1)

    # labels: train only the answer portion; others = -100
    labels = input_ids_full.clone()
    labels[:, :L_prompt] = -100

    # Since num_vision_tokens are prepended, shift labels right by num_vision_tokens
    labels_padded = torch.full(
        (1, L_full + num_vision_tokens),
        -100,
        dtype=torch.long,
        device=device,
    )
    labels_padded[:, num_vision_tokens:] = labels  # first num_vision_tokens positions are -100

    return input_ids_full, labels_padded, L_full

# ======================================================
# 10. Training hyperparameters
# ======================================================
NUM_EPOCHS = 15      # Increase for longer training, e.g., 10/20
LR = 1e-4

optimizer = torch.optim.AdamW(projector.parameters(), lr=LR)

# Directory to save weights
save_dir = "CLIP+tinyllama+multitoken"
os.makedirs(save_dir, exist_ok=True)

# ======================================================
# 11. Evaluation: compute F1 / EM on a dataset
# ======================================================
def evaluate_model(eval_ds):
    projector.eval()
    llama.eval()

    all_f1, all_em = [], []

    for sample in tqdm(eval_ds, desc="Evaluating", leave=False):
        question = sample["question"]
        answer_gt = sample["answer"]
        img = sample["image"]       # PIL Image

        # Image path via CLIP with multiple visual tokens
        with torch.no_grad():
            vision_embeds = extract_vision_embeds(img)  # (1, NUM_VISION_TOKENS, H)

            # Text prompt: question only
            prompt = f"Question: {question}\nAnswer:"
            tok = tokenizer(prompt, return_tensors="pt")
            input_ids = tok.input_ids.to(device)

            text_emb = llama.model.embed_tokens(input_ids)  # (1, L, H)

            # Concatenate N visual tokens + text tokens
            inputs_embeds = torch.cat(
                [vision_embeds, text_emb],
                dim=1
            )

            # Generate answer
            output = llama.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=32,
                do_sample=False,
            )

            pred = tokenizer.decode(output[0], skip_special_tokens=True)
            # Take only the first line after "Answer:"
            pred = pred.split("Answer:")[-1].strip().split("\n")[0].strip()

        all_f1.append(f1(pred, answer_gt))
        all_em.append(em(pred, answer_gt))

    mean_f1 = float(np.mean(all_f1))
    mean_em = float(np.mean(all_em))
    return mean_f1, mean_em

# ======================================================
# 12. Training loop (per epoch: evaluate on 1k val + save weights)
# ======================================================
epoch_history = []  # store results of each epoch

for epoch in range(NUM_EPOCHS):
    projector.train()
    llama.train()    # keep training mode even though params are frozen

    total_loss = 0.0

    # Iterate over the entire training set (one epoch)
    for sample in tqdm(train_ds, desc=f"Epoch {epoch+1} / {NUM_EPOCHS}"):
        question = sample["question"]
        answer_gt = sample["answer"]
        img = sample["image"]

        # Image branch: PIL → CLIP → multiple visual tokens
        vision_embeds = extract_vision_embeds(img)  # (1, NUM_VISION_TOKENS, H)

        # Text + labels
        input_ids, labels_padded, L_full = build_inputs_and_labels(
            question, answer_gt, NUM_VISION_TOKENS
        )

        # Text embeddings (no grad on embeddings)
        with torch.no_grad():
            text_emb = llama.model.embed_tokens(input_ids)  # (1, L_full, H)

        # Concatenate visual tokens + text tokens
        inputs_embeds = torch.cat(
            [vision_embeds, text_emb],
            dim=1
        )  # (1, NUM_VISION_TOKENS + L_full, H)

        # Forward + loss
        outputs = llama(
            inputs_embeds=inputs_embeds,
            labels=labels_padded,
        )
        loss = outputs.loss

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # After each epoch
    avg_loss = total_loss / len(train_ds)
    print(f"\n[Epoch {epoch+1}] train loss = {avg_loss:.4f}")

    # Evaluate on val_ds (up to 1000 samples)
    val_f1, val_em = evaluate_model(val_ds)
    print(f"[Epoch {epoch+1}] val(1k) F1 = {val_f1:.4f}, val(1k) EM = {val_em:.4f}\n")

    # Record this epoch's results
    epoch_history.append({
        "epoch": epoch + 1,
        "train_loss": float(avg_loss),
        "val_f1": float(val_f1),
        "val_em": float(val_em),
    })

    # Save projector weights at the end of each epoch
    save_path = os.path.join(save_dir, f"projector_epoch{epoch+1}.pt")
    torch.save(projector.state_dict(), save_path)
    print(f"Saved projector weights to: {save_path}")

print("Training finished.")

print("Epoch history:")
for record in epoch_history:
    print(record)

# ======================================================
# 13. After training: final evaluation on the full test set
# ======================================================
final_f1, final_em = evaluate_model(test_ds)
print("\n===== Final evaluation on FULL test set =====")
print(f"Test Samples: {len(test_ds)}")
print(f"Final F1: {final_f1:.4f}")
print(f"Final EM: {final_em:.4f}")
