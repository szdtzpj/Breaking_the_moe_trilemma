import time, json, os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Iterator

class SimpleLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.events: List[Dict[str, Any]] = []
        self.filename = "training_log.jsonl"

    def log(self, **kwargs):
        entry = {"t": time.time()}
        entry.update(kwargs)
        self.events.append(entry)

    def dump(self, filename=None):
        if filename is None:
            filename = self.filename
        path = os.path.join(self.log_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"[Logger] Dumped {len(self.events)} events -> {path}")

    @staticmethod
    def load_events(path: str) -> List[Dict[str, Any]]:
        events = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
        return events