import os
import sys
sys.path.append(".")
from pathlib import Path
import torch
import json
import jsonlines
from argparse import ArgumentParser
import copy
from tqdm import tqdm
from inference import Assistant
from train_sft import IGNORE_INDEX, DataCollatorForSupervisedDataset



def process(example, tokenizer):
    processed = []
    user = tokenizer.user_token_id
    assistant = tokenizer.assistant_token_id
    eot = tokenizer.eot_token_id

    def tokenize(s):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s.strip()))

    for choice in example["choices"]:
        input_ids = []
        labels = []
        messages = copy.deepcopy(example["messages"])[:-1]
        for message in messages:
            input_ids.append(user if message["role"] == "user" else assistant)
            labels.append(IGNORE_INDEX)
            content = tokenize(message["content"]) + [eot]
            input_ids.extend(content)
            labels.extend([IGNORE_INDEX] * len(content))
        input_ids.append(assistant)
        labels.append(IGNORE_INDEX)
        content = tokenize(choice) + [eot]
        input_ids.extend(content)
        labels.extend(content)
        input_ids = input_ids[:2048]
        labels = labels[:2048]
        assert len(input_ids) == len(labels)
        attention_mask = [1] * len(input_ids)
        processed.append({'input_ids': torch.LongTensor(input_ids), 'labels': torch.LongTensor(labels),
                          'attention_mask': torch.LongTensor(attention_mask)})

    return processed


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="runs/runs-7b/curated-160/20231017-2215/checkpoint-33"
    )
    parser.add_argument(
        "--eval_data_path", type=str, default="data/curated/valid"
    )
    args = parser.parse_args()

    assistant = Assistant(args.model_name_or_path)

    path = Path(args.eval_data_path)
    data_files = [os.path.join(path, file.name) for file in path.glob("*.json")]
    for data_file in data_files:
        dir_name = os.path.dirname(data_file)
        file_name = os.path.basename(data_file)
        input_path = os.path.join(dir_name, file_name)
        base, ckpname = os.path.split(args.model_name_or_path)
        base, timestamp = os.path.split(base)
        base, model_type = os.path.split(base)
        base, model_sz = os.path.split(base)
        assert "runs-" in model_sz
        model_sz = model_sz.replace("runs-", "")
        output_path = os.path.join("evaluate", "pred-data", os.path.split(dir_name)[-1], model_sz, model_type, ckpname, "pred_"+file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        data = []
        with open(input_path, 'r', encoding='utf8') as f:
            for line in f:
                data.append(json.loads(line))

        for sample in tqdm(data):
            if sample["question_format"] == 0:
                test_sample = copy.deepcopy(sample)
                test_sample["messages"] = test_sample["messages"][:-1]
                responses, scores = assistant.inference([test_sample])
                generated_response = responses[0] # string
                generated_score = scores[0].tolist() # |V|
                assert "generated_response" not in sample
                sample["generated_response"] = generated_response
                assert "generated_score" not in sample
                sample["generated_score"] = generated_score

                with jsonlines.open(output_path, mode="a") as f:
                    f.write(sample)

            elif sample["question_format"] == 1:
                assert "choices" in sample
                assert len(sample["choices"]) == 3
                assert "generated_ppl" not in sample

                tokenizer = assistant.tokenizer
                model = assistant.model
                processed_samples = process(sample, tokenizer)
                assert len(processed_samples) == 3
                data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, pad_to_multiple_of=8)
                generated_ppl = []
                for processed_sample in processed_samples:
                    inputs = data_collator([processed_sample])

                    outputs = model(**inputs)
                    for b in range(outputs.logits.size()[0]):
                        logits = outputs.logits[b][..., :-1, :].contiguous()
                        labels = inputs['labels'][b][..., 1:].contiguous()
                        cel = torch.nn.CrossEntropyLoss()
                        lm_loss = cel(logits, labels)
                        generated_ppl.append( torch.exp(lm_loss).item() )

                assert len(generated_ppl) == 3
                sample["generated_ppl"] = generated_ppl

                with jsonlines.open(output_path, mode="a") as f:
                    f.write(sample)

            else:
                raise ValueError("Invalid question_format!!!")
        print("Done!")

if __name__ == "__main__":
    with torch.no_grad():
        main()
