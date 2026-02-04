from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

def build_lm_loaders(config, distributed=False):
    ds = load_dataset(config["dataset_name"], config["subset"])
    tokenizer = AutoTokenizer.from_pretrained(config.get("vocab_name","bert-base-uncased"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token
    block_size = config["block_size"]
    batch_size = config["batch_size"]

    def tokenize(example):
        return tokenizer(example["text"])
    tokenized = ds["train"].map(tokenize, batched=True, remove_columns=ds["train"].column_names)

    def group_texts(examples):
        ids = sum(examples["input_ids"], [])
        total_len = (len(ids) // block_size) * block_size
        ids = ids[:total_len]
        chunks = [ids[i:i+block_size] for i in range(0, total_len, block_size)]
        return {"input_ids": chunks}
    lm_train = tokenized.map(group_texts, batched=True)

    tokenized_val = ds["validation"].map(tokenize, batched=True, remove_columns=ds["validation"].column_names)
    lm_val = tokenized_val.map(group_texts, batched=True)

    lm_train.set_format(type="torch", columns=["input_ids"])
    lm_val.set_format(type="torch", columns=["input_ids"])

    sampler_train = sampler_val = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler_train = DistributedSampler(lm_train, shuffle=config.get("shuffle",True))
        sampler_val = DistributedSampler(lm_val, shuffle=False)

    def collate(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask, labels

    train_loader = DataLoader(lm_train, batch_size=batch_size,
                              shuffle=(sampler_train is None and config.get("shuffle", True)),
                              sampler=sampler_train,
                              num_workers=config.get("num_workers",0),
                              collate_fn=collate)
    val_loader = DataLoader(lm_val, batch_size=batch_size,
                            shuffle=False, sampler=sampler_val,
                            num_workers=config.get("num_workers",0),
                            collate_fn=collate)
    return train_loader, val_loader, tokenizer