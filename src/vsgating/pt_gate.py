import os

from torch import nn
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  # speedrun trick #1

import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, HfArgumentParser, DataCollatorForLanguageModeling, AutoTokenizer
import sys

import datasets
import wandb

from vsgating.modeling_gating import GateLM, GateConfig

datasets.config.IN_MEMORY_MAX_SIZE = 500 * 1024 * 1024  
torch.set_float32_matmul_precision("high")  # speedrun trick — free tensor-core throughput


def train(training_args: TrainingArguments):
    # Load the dataset
    train_ds = load_dataset("avgJo3/fineweb-subset-100M", split="train")
    eval_ds  = load_dataset("avgJo3/fineweb-subset-100M", split="eval")

    use_cuda = torch.cuda.is_available()

    config = GateConfig(
        d_model=512,
        num_heads=8,
        num_layers=12,
        vocab_size=50257,
        scale=4,
        device="cuda" if use_cuda else "cpu",
    )
    model = GateLM(config)
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            m.weight.data = m.weight.data.bfloat16()    

    print(model)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    assert tokenizer.eos_token_id is not None, "eos_token_id is None — tokenizer failed to load special tokens"
    assert tokenizer.pad_token_id is not None, "pad_token_id still None after assignment"
    print("eos_token:", tokenizer.eos_token, "| eos_token_id:", tokenizer.eos_token_id)
    print("pad_token:", tokenizer.pad_token, "| pad_token_id:", tokenizer.pad_token_id)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )
    try:
        trainer.train()
        # trainer.save_model(training_args.output_dir)
    finally:
        wandb.finish()    





def main():
    parser = HfArgumentParser(TrainingArguments)

    default_args = [
        "--output_dir", "/content/model/pt-gate",
        "--do_train", "True",
        "--do_eval", "True",
        "--per_device_train_batch_size", "2",
        "--per_device_eval_batch_size", "2",
        "--gradient_accumulation_steps", "1",
        "--num_train_epochs", "1",
        "--logging_strategy", "steps",
        "--logging_steps", "5",
        "--eval_strategy", "steps",
        "--eval_steps", "50",
        "--eval_on_start", "True",
        "--save_strategy", "no",
        "--seed", "42",
        "--bf16", str(torch.cuda.is_available()),


        "--dataloader_num_workers", "8",

        "--report_to", "wandb",
        "--max_grad_norm", "1.0",  

        "--optim", "adamw_torch_fused",
        "--learning_rate",       "3e-4",      # peak lr → min will be 3e-5
        "--lr_scheduler_type",   "cosine_with_min_lr",
        "--lr_scheduler_kwargs", '{"min_lr_rate": 0.1}',
        "--warmup_ratio",        "0.01",      # 1% of total steps
        "--weight_decay",        "0.1",
        "--adam_beta2",          "0.95",        
    ]



    (training_args,) = parser.parse_args_into_dataclasses(
        args=default_args + sys.argv[1:]
    )

    train(training_args)


if __name__ == "__main__":
    main()