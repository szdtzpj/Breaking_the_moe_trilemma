import yaml, pprint, sys
with open("configs_training_config_mini_Version3.yaml","r",encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print("Top-level keys:", cfg.keys())
print("model section:", cfg.get("model"))
print("model.moe exists? ->", isinstance(cfg.get("model"), dict) and "moe" in cfg["model"])
if "model" in cfg and "moe" not in cfg["model"]:
    print("Keys under model:", list(cfg["model"].keys()))

    python -B scripts_run_classification_Version2.py --config configs_training_config_8g_classification_Version2.yaml