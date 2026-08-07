import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, HfArgumentParser, DataCollatorForLanguageModeling, AutoTokenizer
import sys

from vsgating.modeling_ref_v2 import RefLM, RefConfig

import datasets
datasets.config.IN_MEMORY_MAX_SIZE = 500 * 1024 * 1024  

def train(training_args: TrainingArguments):
    # Load the dataset
    train_ds = load_dataset("avgJo3/fineweb-subset-10M", split="train")
    eval_ds  = load_dataset("avgJo3/fineweb-subset-10M", split="eval")

    use_cuda = torch.cuda.is_available()

    config = RefConfig(
        d_model=64,
        num_heads=4,
        num_layers=4,
        vocab_size=50257,  # GPT-2 tokenizer vocab; swap for your tokenizer's size
        scale=4,           # MLP hidden-size multiplier (d_model * scale)
        device="cuda" if use_cuda else "cpu",
    )
    model = RefLM(config)

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
    trainer.train()
    # trainer.save_model(training_args.output_dir)



def main():
    parser = HfArgumentParser(TrainingArguments)

    default_args = [
        "--output_dir", "./checkpoints/pretrain-proto",
        "--do_train", "True",
        "--do_eval", "True",
        "--per_device_train_batch_size", "64",
        "--per_device_eval_batch_size", "64",
        "--gradient_accumulation_steps", "1",
        "--num_train_epochs", "1",
        "--learning_rate", "3e-4",
        "--weight_decay", "0.01",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_strategy", "steps",
        "--logging_steps", "5",
        "--eval_strategy", "steps",
        "--eval_steps", "10",
        "--eval_on_start", "True",
        "--save_strategy", "no",
        "--seed", "42",
        "--bf16", str(torch.cuda.is_available()),
        "--dataloader_num_workers", "8",
        "--report_to", "wandb",
    ]

    (training_args,) = parser.parse_args_into_dataclasses(
        args=default_args + sys.argv[1:]
    )

    train(training_args)


if __name__ == "__main__":
    main()
