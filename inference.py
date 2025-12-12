import torch
from PIL import Image
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPVisionModel,
    CLIPImageProcessor,
)

# === Hyperparameters (Must match training) ===
NUM_VISION_TOKENS = 4

# === Projector Definition (Must match training) ===
class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

# === Pluggable Inference Function ===
def infer(image, question, projector_ckpt,
          clip_name="openai/clip-vit-base-patch32",
          llama_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
          device=None):

    # 1. Device Setup
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load CLIP Vision Model (Frozen)
    clip_vision = CLIPVisionModel.from_pretrained(clip_name).to(device)
    clip_processor = CLIPImageProcessor.from_pretrained(clip_name)
    clip_vision.eval()
    for p in clip_vision.parameters():
        p.requires_grad = False

    vision_width = clip_vision.config.hidden_size  # e.g., 768

    # 3. Load TinyLLaMA (Frozen)
    tokenizer = AutoTokenizer.from_pretrained(llama_name)
    llama = AutoModelForCausalLM.from_pretrained(
        llama_name,
        torch_dtype=torch.float16,
    ).to(device)
    llama.eval()
    for p in llama.parameters():
        p.requires_grad = False

    lm_hidden_size = llama.config.hidden_size  # e.g., 2048

    # 4. Load Projector
    projector = Projector(vision_width, lm_hidden_size).to(device)
    try:
        projector.load_state_dict(torch.load(projector_ckpt, map_location=device))
    except Exception as e:
        raise ValueError(f"Failed to load projector checkpoint from {projector_ckpt}: {e}")
        
    projector.eval()

    # 5. Process Image Input
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    clip_inputs = clip_processor(images=image, return_tensors="pt").to(device)

    # 6. Extract Vision Features (New 4-token Logic)
    with torch.no_grad():
        vision_outputs = clip_vision(**clip_inputs)
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]  # remove CLS (B, S, C)
        
        B, S, C = patch_tokens.shape
        num_tokens = NUM_VISION_TOKENS

        # Evenly divide into N groups
        if S < num_tokens:
            repeat_factor = (num_tokens + S - 1) // S
            patch_tokens = patch_tokens.repeat(1, repeat_factor, 1)
            S = patch_tokens.size(1)

        chunk_size = S // num_tokens
        usable_len = chunk_size * num_tokens
        patch_tokens = patch_tokens[:, :usable_len, :]
        patch_tokens = patch_tokens.view(B, num_tokens, chunk_size, C)
        patch_tokens = patch_tokens.mean(dim=2)  # (B, NUM_TOKENS, C)

        embeds_f32 = projector(patch_tokens)
        vision_embeds = embeds_f32.to(torch.float16)  # (B, NUM_TOKENS, LM_HIDDEN)

    # 7. Construct Text Prompt Embedding
    prompt = f"Question: {question}\nAnswer:"
    tok = tokenizer(prompt, return_tensors="pt")
    input_ids = tok.input_ids.to(device)

    # Convert text tokens to embeddings
    text_emb = llama.get_input_embeddings()(input_ids)  # (B, L, LM_HIDDEN)

    # 8. Concatenate Vision + Text Embeddings
    # vision_embeds: [B, NUM_VISION_TOKENS, H]
    # text_emb: [B, Text_Len, H]
    inputs_embeds = torch.cat([vision_embeds, text_emb], dim=1)

    # 9. Generate Answer
    output_ids = llama.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=32, # Aligned with notebook
        do_sample=False,   # Aligned with notebook
        pad_token_id=tokenizer.eos_token_id
    )

    # 10. Format Output
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "Answer:" in text:
        ans = text.split("Answer:", 1)[-1].strip().split("\n")[0]
    else:
        ans = text.strip()
    
    # Clean up repetitive output
    words = ans.split()
    if len(words) > 3:
        first_few = ' '.join(words[:3])
        if ans.count(first_few) > 1:
            ans = ans.split(first_few)[0] + first_few
            ans = ans.strip()
    
    # Remove trailing incomplete sentences
    for punct in ['.', '!', '?']:
        if punct in ans:
            last_idx = ans.rfind(punct)
            if last_idx > 0:
                ans = ans[:last_idx + 1]
                break
    
    print(f"[DEBUG] Raw Answer: {ans}")
    if len(ans) < 1:
        ans = "Unable to analyze the image clearly."

    return ans


# === Independent Run Example ===
if __name__ == "__main__":
    import sys
    import os
    
    print("=== Medical Visual Q&A Inference Module (New Model) ===")
    print("\nUsage Example:")
    print("python inference.py <image_path> <question> [projector_ckpt]")
    print("\nArguments:")
    print("  image_path: Path to image file")
    print("  question: Question about the image")
    print("  projector_ckpt: (Optional) Path to projector checkpoint")
    
    if len(sys.argv) >= 3:
        image_path = sys.argv[1]
        question = sys.argv[2]
        projector_ckpt = sys.argv[3] if len(sys.argv) > 3 else "newlytrained/projector_epoch1.pt"
        
        # Check if default checkpoint exists, fallback to local if needed
        if not os.path.exists(projector_ckpt) and os.path.exists("projector_epoch1.pt"):
             projector_ckpt = "projector_epoch1.pt"

        if not os.path.exists(projector_ckpt):
            print(f"\n❌ Error: Projector checkpoint not found: {projector_ckpt}")
            sys.exit(1)
        
        if not os.path.exists(image_path):
            print(f"\n❌ Error: Image file not found: {image_path}")
            sys.exit(1)
        
        print(f"\n🖼️  Image: {image_path}")
        print(f"❓ Question: {question}")
        print(f"📦 Model: {projector_ckpt}")
        print("\n⏳ Running Inference...")
        
        try:
            answer = infer(
                image=image_path,
                question=question,
                projector_ckpt=projector_ckpt
            )
            print(f"\n✅ Answer: {answer}\n")
        except Exception as e:
            print(f"\n❌ Inference Failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
