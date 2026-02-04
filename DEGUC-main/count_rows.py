from deguc.data.classification_dataset import build_classification_loaders
cfg = {
  "local_path":"data/glue_sst2_local",
  "dataset_name":"glue","subset":"sst2",
  "max_seq_len":64,"batch_size":32,
  "shuffle":True,"num_workers":0,
  "vocab_name":"bert-base-uncased"
}
tl, vl, te, tok = build_classification_loaders(cfg)
print("Lens (train,val,test):", len(tl), len(vl), len(te))
first = next(iter(tl))
print("First batch shapes:", first[0].shape, first[1].shape, first[2].shape)