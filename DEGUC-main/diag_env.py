# save as diag_env.py
import os, datasets, huggingface_hub, requests

print("datasets version:", datasets.__version__)
print("huggingface_hub version:", huggingface_hub.__version__)
print("HF_ENDPOINT =", os.environ.get("HF_ENDPOINT"))
print("HF_DATASETS_OFFLINE =", os.environ.get("HF_DATASETS_OFFLINE"))
print("HTTP_PROXY =", os.environ.get("HTTP_PROXY"))
print("HTTPS_PROXY =", os.environ.get("HTTPS_PROXY"))

# test
for url in ["https://huggingface.co", "https://huggingface.co/datasets/glue"]:
    try:
        r = requests.get(url, timeout=5)
        print(url, "status", r.status_code)
    except Exception as e:
        print(url, "ERROR", e)

# try load
try:
    ds = datasets.load_dataset("glue","sst2")
    print("Loaded GLUE SST2 successfully:", ds)
except Exception as e:

    print("Load GLUE failed:", repr(e))
