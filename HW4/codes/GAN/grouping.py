import json
import re
from collections import defaultdict
import numpy as np

if __name__ == "__main__":
    with open('log.jsonl', 'r') as f:
        logs = f.read().strip().split("\n")
    logs = map(json.loads, logs)
    runs = defaultdict(lambda: [[], []])
    for log in logs:
        config, seed = re.search(r"(.*?)_SD(\d{4})", log['name']).groups()
        runs[config][0].append(log['fid'])
        runs[config][1].append(seed)
    print(runs)
    runs = {k: (np.array(v), sd) for k, (v, sd) in runs.items()}
    for k, (v, sd) in runs.items():
        L, G = re.search("z-(L\d+)_(G[\s\S]+)_D", k).groups()
        print("%s %s %.01f ± %.01f" % (
            L, G, v.mean(), v.std()
        ))
        print(" > %s %s %.01f" % (
            k, sd[np.argmin(v)], v.min()
        ))


