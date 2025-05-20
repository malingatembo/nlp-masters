import os
import re

patterns = [
    r'--PARENT',
    r'--ID',
    r'\[\[CODE:',
    r'\[\[TABLE:',
    r'<!--',
    r'-->',
    r'^###\s'
]
compiled = [re.compile(p) for p in patterns]
root = "."

matched_files = set()

for dirpath, _, filenames in os.walk(root):
    for fname in filenames:
        if fname.endswith(".json"):
            path = os.path.join(dirpath, fname)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                if any(p.search(content) for p in compiled):
                    matched_files.add(path)

for f in sorted(matched_files):
    print(f)
