from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, BertTokenizerFast, PreTrainedTokenizerFast
import torch
import os

# top collate，can pickled by Windows
def classification_collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.tensor([int(b["label"]) for b in batch])
    return input_ids, attention_mask, labels

def build_classification_loaders(config, distributed=False):
    local_path = config.get("local_path", None)
    if not local_path:
        raise ValueError("FORCED LOCAL MODE: Missing local_path.")

    data_files = {
        "train": os.path.join(local_path, "train.tsv"),
        "validation": os.path.join(local_path, "validation.tsv"),
        "test": os.path.join(local_path, "test.tsv"),
    }
    print(">>> USING LOCAL DATA:", data_files)
    for k,p in data_files.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"[LOCAL DATA] Missing {k} split: {p}")

    ds = load_dataset("csv", data_files=data_files, delimiter="\t")

    if "label" not in ds["train"].column_names:
        raise ValueError("train.tsv is missing the label column")
    def force_int(e):
        e["label"] = int(e["label"])
        return e
    ds = ds.map(force_int)

    vocab_name = config.get("vocab_name", "bert-base-uncased")
    if os.path.isdir(vocab_name):
        tok_json = os.path.join(vocab_name, "tokenizer.json")
        vocab_txt = os.path.join(vocab_name, "vocab.txt")
        if os.path.isfile(tok_json):
            print(f">>> USING LOCAL TRAINED TOKENIZER: {vocab_name}")
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=tok_json,
                unk_token="[UNK]", pad_token="[PAD]",
                cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]"
            )
        elif os.path.isfile(vocab_txt):
            print(f">>> USING LOCAL VOCAB.TXT TOKENIZER: {vocab_name}")
            tokenizer = BertTokenizerFast(vocab_file=vocab_txt, do_lower_case=True)
        else:
            print(f"[WARN] {vocab_name} does not contain tokenizer.json / vocab.txt, attempting online loading (may fail)")
            tokenizer = AutoTokenizer.from_pretrained(vocab_name)
    else:
        # If online loading is allowed, this line can be kept; to be completely offline, change this line to raise
        tokenizer = AutoTokenizer.from_pretrained(vocab_name)

    max_len = config["max_seq_len"]
    batch_size = config["batch_size"]

    def encode(batch):
        enc = tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_len
        )
        batch["input_ids"] = enc["input_ids"]
        batch["attention_mask"] = enc["attention_mask"]
        return batch

    ds = ds.map(encode, batched=True)
    ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])

    train_ds = ds["train"]
    val_ds = ds["validation"]
    test_ds = ds["test"]

    sampler_train = sampler_val = sampler_test = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler_train = DistributedSampler(train_ds, shuffle=config.get("shuffle", True))
        sampler_val = DistributedSampler(val_ds, shuffle=False)
        sampler_test = DistributedSampler(test_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler_train is None and config.get("shuffle", True)),
        sampler=sampler_train,
        num_workers=config.get("num_workers",0),
        collate_fn=classification_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, sampler=sampler_val,
        num_workers=config.get("num_workers",0),
        collate_fn=classification_collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, sampler=sampler_test,
        num_workers=config.get("num_workers",0),
        collate_fn=classification_collate
    )
    return train_loader, val_loader, test_loader, tokenizer
