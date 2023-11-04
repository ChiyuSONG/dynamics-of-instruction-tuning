# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

def read_jsonl(path):
    with open(path, encoding='utf8') as fh:
        results = []
        for line in fh:
            if line is None:
                continue
            try:
                results.append(json.loads(line) if line != "null" else line)
            except Exception as e:
                print(e)
                print(path)
                print(line)
                raise e
    return results

def save_jsonl(lines, directory):
    with open(directory, 'w', encoding='utf8') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")