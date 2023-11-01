import os
import sys
sys.path.append(".")
from pathlib import Path
import json
import jsonlines
from argparse import ArgumentParser
from tqdm import tqdm
from utils import *
from transformers import (
    LlamaTokenizer,
)

# tokenizer from any trained checkpoints is fine
tokenizer = LlamaTokenizer.from_pretrained("models/eval-tokenizer")

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--pred_data_path", type=str, default="evaluate/pred-data/test"
    )
    args = parser.parse_args()

    root_path = Path(args.pred_data_path)
    sz_dirs = os.listdir(root_path)
    sz_dirs = [sz_dir for sz_dir in sz_dirs if os.path.isdir(os.path.join(root_path, sz_dir))]
    for sz_dir in sz_dirs:
        sz_path = os.path.join(root_path, sz_dir)
        runs_dirs = os.listdir(sz_path)
        runs_dirs = [runs_dir for runs_dir in runs_dirs if os.path.isdir(os.path.join(sz_path, runs_dir))]
        for runs_dir in runs_dirs:
            runs_path = os.path.join(sz_path, runs_dir)
            ckps_dirs = os.listdir(runs_path)
            ckps_dirs = [ckps_dir for ckps_dir in ckps_dirs if
                         os.path.isdir(os.path.join(runs_path, ckps_dir))]
            for i, ckps_dir in enumerate(ckps_dirs):
                path = Path(os.path.join(runs_path, ckps_dir))
                base, ckp_name = os.path.split(path)
                base, model_type = os.path.split(base)
                base, model_sz = os.path.split(base)
                base, dataset = os.path.split(base)

                output_path = os.path.join("evaluate", "results", dataset, model_sz, model_type,
                                           ckp_name + "-results.json")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                data_files = [os.path.join(path, file.name) for file in path.glob("*.json")]
                scores = {}
                for data_file in data_files:
                    dir_name = os.path.dirname(data_file)
                    file_name = os.path.basename(data_file)
                    input_path = os.path.join(dir_name, file_name)

                    data = []
                    with open(input_path, 'r', encoding='utf8') as f:
                        for line in f:
                            data.append(json.loads(line))

                    score = 0
                    score_expected = 0
                    for sample in tqdm(data):
                        if sample["question_format"] == 0:
                            if sample["type"] == "math":
                                sample_score = math_score(sample, tokenizer=tokenizer)
                                score += sample_score["score"]
                                score_expected += sample_score["score_expected"]
                            elif sample["type"] == "ethics":
                                sample_score = ethics_score(sample, tokenizer=tokenizer)
                                score += sample_score["score"]
                                score_expected += sample_score["score_expected"]
                            elif sample["type"] == "understanding":
                                sample_score = understanding_score(sample, tokenizer=tokenizer)
                                score += sample_score["score"]
                                score_expected += sample_score["score_expected"]
                            elif sample["type"] == "reasoning":
                                sample_score = reasoning_score(sample, tokenizer=tokenizer)
                                score += sample_score["score"]
                                score_expected += sample_score["score_expected"]
                            elif sample["type"] == "history":
                                sample_score = history_score(sample, tokenizer=tokenizer)
                                score += sample_score["score"]
                                score_expected += sample_score["score_expected"]
                            elif sample["type"] == "biology":
                                sample_score = biology_score(sample, tokenizer=tokenizer)
                                score += sample_score["score"]
                                score_expected += sample_score["score_expected"]
                            elif sample["type"] == "chinese":
                                sample_score = chinese_score(sample, tokenizer=tokenizer)
                                score += sample_score["score"]
                                score_expected += sample_score["score_expected"]
                            elif sample["type"] == "ceval_physician_fshot":
                                sample_score = ceval_physician_fshot_score(sample, tokenizer=tokenizer)
                                score += (sample_score["score"] /40 * 100)
                                score_expected += (sample_score["score_expected"] /40.0 * 100)
                            elif sample["type"] == "ceval_teacher_qualification_fshot":
                                sample_score = ceval_teacher_qualification_fshot_score(sample, tokenizer=tokenizer)
                                score += (sample_score["score"] /40.0 * 100)
                                score_expected += (sample_score["score_expected"] /40.0 * 100)
                            elif sample["type"] == "ceval_urban_and_rural_planner_fshot":
                                sample_score = ceval_urban_and_rural_planner_fshot_score(sample, tokenizer=tokenizer)
                                score += (sample_score["score"] /40.0 * 100)
                                score_expected += (sample_score["score_expected"] /40.0 * 100)
                            else:
                                raise ValueError

                        elif sample["question_format"] == 1:
                            generated_ppl = sample["generated_ppl"]

                            min_ppl = min(generated_ppl)
                            if min_ppl == generated_ppl[0]:
                                score += 1

                            score_expected += 1.0 * (1.0 / 3)


                    scores[file_name] = score
                    scores["expected_" + file_name] = score_expected

                l = len(scores)
                scores["average"] = sum([scores[k] for k in scores if "expected" not in k]) / float(l)
                scores["expected_average"] = sum([scores[k] for k in scores if "expected" in k]) / float(l)
                with jsonlines.open(output_path, mode="w") as f:
                    for key in scores:
                        f.write({key: scores[key]})

    print("Done!")

if __name__ == "__main__":
    main()
