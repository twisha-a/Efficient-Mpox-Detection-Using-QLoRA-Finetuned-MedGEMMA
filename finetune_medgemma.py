#!/usr/bin/env python3
import os, json, logging, argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from datasets import load_dataset, ClassLabel
from datasets import Features, Value, ClassLabel
# from transformers import (
#     AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# )
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from evaluate import load as load_metric

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("finetune_medgemma_mpox")

# ---------------- Task prompt ----------------
CLASSES = ["A: mpox", "B: not mpox"]
PROMPT = "What is the most likely diagnosis for the skin lesion in the image?\n" + "\n".join(CLASSES)

# ---------------- Helpers ----------------
def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        sm = torch.cuda.get_device_capability()[0]
        if sm >= 8:
            log.info("Using bfloat16 (GPU SM %d)", sm)
            return torch.bfloat16
        else:
            log.warning("GPU SM < 80 — using float16")
            return torch.float16
    log.warning("No CUDA detected, using float32 (CPU)")
    return torch.float32

def build_bnb(dtype: torch.dtype) -> BitsAndBytesConfig:
    # prefer bf16 compute when available
    storage = torch.float16 if dtype == torch.float16 else dtype
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_storage=storage,
    )

def format_train_row(ex: Dict[str, Any]) -> Dict[str, Any]:
    ex["messages"] = [
        {"role":"user","content":[{"type":"image"},{"type":"text","text":PROMPT}]},
        {"role":"assistant","content":[{"type":"text","text": ex["label_text"]}]},
    ]
    return ex

def format_test_row(ex: Dict[str, Any]) -> Dict[str, Any]:
    ex["messages"] = [
        {"role":"user","content":[{"type":"image"},{"type":"text","text":PROMPT}]},
    ]
    return ex

def collate_fn_builder(processor):
    def collate_fn(batch: List[Dict[str, Any]]):
        texts, images = [], []
        for ex in batch:
            ipath = ex["image_path"]
            try:
                img = Image.open(ipath).convert("RGB")
            except Exception as e:
                log.error("Failed to open image: %s | %s", ipath, e)
                img = Image.new("RGB", (1,1), (0,0,0))  # keep batch alive
            images.append([img])
            texts.append(processor.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            ).strip())
        batch_enc = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch_enc["input_ids"].clone()
        # mask pad + image tokens from loss
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        boi_tok = processor.tokenizer.special_tokens_map.get("boi_token", None)
        if boi_tok is not None:
            boi_id = processor.tokenizer.convert_tokens_to_ids(boi_tok)
            if isinstance(boi_id, int):
                labels[labels == boi_id] = -100
        # labels[labels == 262144] = -100  # some Gemma builds place image placeholder here
        for tok in ["<image>", "<image_placeholder>", "<im_start>", "<im_end>"]:
            if tok in processor.tokenizer.get_vocab():
                tid = processor.tokenizer.convert_tokens_to_ids(tok)
                labels[labels == tid] = -100

        batch_enc["labels"] = labels
        return batch_enc
    return collate_fn

def postprocess_builder(label_feature: ClassLabel):
    alt = {lab: f"({lab.replace(': ', ') ')}" for lab in CLASSES}
    def f(pred: List[Dict[str,str]], strict: bool=False) -> int:
        txt = pred[0]["generated_text"]
        if strict:
            try: return label_feature.str2int(txt)
            except Exception: return -1
        for lab in CLASSES:
            if lab in txt or alt[lab] in txt:
                return label_feature.str2int(lab)
        return -1
    return f

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    # ----- DDP rank setup (needed for torchrun) -----
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    ap.add_argument("--model_id", default="google/medgemma-4b-it")
    ap.add_argument("--data_dir", default="mpox_data_prepared")
    ap.add_argument("--output_dir", default="medgemma-4b-it-qlora-mpox")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--train_bs", type=int, default=2)
    ap.add_argument("--eval_bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval_subset", type=int, default=400)
    ap.add_argument("--eval_strict", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_quant", action="store_true", help="Disable 4-bit quant (use bf16/fp16 full-precision)")
    ap.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit instead of 4-bit")
    args = ap.parse_args()

    # ---- Load datasets from JSONL ----
    tr_path = Path(args.data_dir) / "train.jsonl"
    va_path = Path(args.data_dir) / "val.jsonl"
    te_path = Path(args.data_dir) / "test.jsonl"

    
    if not (tr_path.is_file() and va_path.is_file() and te_path.is_file()):
        raise FileNotFoundError("Expected train/val/test JSONL files under --data_dir")

    log.info("Loading JSONL data from %s", args.data_dir)
    features = Features({
    "image_path": Value("string"),
    "label_text": Value("string"),  # <— keep as string
    "label_id": Value("int64"),
    "metadata": Value("string"),
    })

    train = load_dataset("json", data_files=str(tr_path), features=features)["train"]
    val   = load_dataset("json", data_files=str(va_path), features=features)["train"]
    test  = load_dataset("json", data_files=str(te_path), features=features)["train"]

    # attach numeric labels (0/1)
    label_feature = ClassLabel(names=CLASSES)

    def add_label(ex):
        lt = ex.get("label_text")
        li = ex.get("label_id")
        if isinstance(lt, str) and lt:
            ex["label"] = label_feature.str2int(lt)
        elif isinstance(li, int):
            ex["label"] = int(li)
        else:
            # last-resort fallback (rare)
            ex["label"] = -1
        return ex

    train = train.map(add_label)
    val   = val.map(add_label)
    test  = test.map(add_label)

    # format messages
    log.info("Formatting messages")
    train = train.map(format_train_row, desc="format-train")
    val   = val.map(format_train_row, desc="format-val")
    test  = test.map(format_test_row,  desc="format-test")

    # ---- Model & processor ----
    dtype = pick_dtype()
    log.info("Loading processor: %s", args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "right"

    quant_cfg = None
    load_kwargs = dict(attn_implementation="eager", torch_dtype=dtype)

    # In distributed mode, pin the whole model to this rank's GPU.
    if world_size > 1:
        load_kwargs["device_map"] = {"": local_rank}
    else:
        load_kwargs["device_map"] = "auto"

    if not args.no_quant:
        try:
            if args.load_in_8bit:
                log.info("Using 8-bit load (bitsandbytes)")
                quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
            else:
                log.info("Using 4-bit QLoRA load (bitsandbytes)")
                quant_cfg = build_bnb(dtype)
            load_kwargs["quantization_config"] = quant_cfg
        except Exception as e:
            log.warning("bitsandbytes config failed: %s — falling back to no quant", e)


    log.info("Loading model: %s", args.model_id)
    # model = AutoModelForImageTextToText.from_pretrained(args.model_id, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)


    # after model load is fine, keep this (it’s standard for checkpointing)
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    
    # ---- LoRA ----
    log.info("Configuring LoRA adapters")
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    # ---- Training config ----
    log.info("Setting up SFT config")
    sft_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=(dtype==torch.bfloat16),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="steps",
        optim="adamw_torch_fused",
        eval_steps=50,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        push_to_hub=False,
        report_to="tensorboard",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
    )

    collate_fn = collate_fn_builder(processor)
    if args.eval_subset > 0 and len(val) > args.eval_subset:
        val_eval = val.shuffle(seed=args.seed).select(range(args.eval_subset))
    else:
        val_eval = val

    # ---- Trainer ----
    log.info("Initializing trainer")
    trainer = SFTTrainer(
        model=model, args=sft_args,
        train_dataset=train, eval_dataset=val_eval,
        peft_config=peft_cfg,
        processing_class=processor,
        data_collator=collate_fn,
    )

    # ---- Train ----
    log.info("Starting training…")
    trainer.train()
    log.info("Training complete.")

    log.info("Saving model & processor to %s", args.output_dir)
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
#     # ---- Evaluation on test ----
#     log.info("Preparing pipeline for test evaluation")
#     processor.tokenizer.padding_side = "left"
#     gen_pipe = pipeline(
#         "image-text-to-text",
#         model=args.output_dir,
#         tokenizer=processor.tokenizer,
#         feature_extractor=processor.image_processor,
#         torch_dtype=dtype,
#     )
#     gen_pipe.model.generation_config.do_sample = False
#     gen_pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

#     log.info("Rendering prompts for %d test samples", len(test))
#     test_texts, test_imgs = [], []
#     for ex in test:
#         test_texts.append(processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False))
#         try:
#             test_imgs.append(Image.open(ex["image_path"]).convert("RGB"))
#         except Exception as e:
#             log.error("Test image open failed: %s | %s", ex["image_path"], e)
#             test_imgs.append(Image.new("RGB", (1,1), (0,0,0)))

#     log.info("Running generation …")
#     batch_size = max(1, min(64, args.eval_bs * 8))
#     outs = gen_pipe(text=test_texts, images=test_imgs, max_new_tokens=20, batch_size=batch_size, return_full_text=False)

#     post = postprocess_builder(label_feature)
#     preds = [post([o], strict=args.eval_strict) for o in outs]
#     refs  = list(test["label"])

#     # filter invalid -1
#     valid = [(p,r) for p,r in zip(preds, refs) if p != -1]
#     if not valid:
#         log.error("All predictions invalid; cannot compute metrics")
#         return
#     p_ok, r_ok = zip(*valid)

#     acc = load_metric("accuracy").compute(predictions=p_ok, references=r_ok)
#     f1w = load_metric("f1").compute(predictions=p_ok, references=r_ok, average="weighted")
#     log.info("TEST RESULTS | accuracy=%.4f | f1_weighted=%.4f", acc["accuracy"], f1w["f1"])

# if __name__ == "__main__":
#     main()


# ---- Evaluation on test ----
    log.info("Running test evaluation with model.generate()")
    model.eval()

    # Ensure pad token id is set
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    test_texts, test_imgs = [], []
    for ex in test:
        test_texts.append(
            processor.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=True
            ).strip()
        )
        try:
            test_imgs.append(Image.open(ex["image_path"]).convert("RGB"))
        except Exception as e:
            log.error("Test image open failed: %s | %s", ex["image_path"], e)
            test_imgs.append(Image.new("RGB", (1, 1), (0, 0, 0)))

    post = postprocess_builder(label_feature)
    preds = []

    eval_bs = max(1, args.eval_bs)

    with torch.no_grad():
        for i in range(0, len(test_texts), eval_bs):
            bt = test_texts[i : i + eval_bs]
            bi = test_imgs[i : i + eval_bs]

            enc = processor(text=bt, images=bi, return_tensors="pt", padding=True)

            device = model.device
            for k, v in enc.items():
                if torch.is_tensor(v):
                    enc[k] = v.to(device)

            gen_ids = model.generate(
                **enc,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

            gen_texts = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            for t in gen_texts:
                preds.append(post([{"generated_text": t}], strict=args.eval_strict))

    refs = list(test["label"])

    valid = [(p, r) for p, r in zip(preds, refs) if p != -1]
    if not valid:
        log.error("All predictions invalid; cannot compute metrics")
        return
    p_ok, r_ok = zip(*valid)

    acc = load_metric("accuracy").compute(predictions=p_ok, references=r_ok)
    f1w = load_metric("f1").compute(predictions=p_ok, references=r_ok, average="weighted")
    log.info("TEST RESULTS | accuracy=%.4f | f1_weighted=%.4f", acc["accuracy"], f1w["f1"])

if __name__ == "__main__":
    main()








