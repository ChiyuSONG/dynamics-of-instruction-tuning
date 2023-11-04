# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer
from src import utils, dataset_loader

IGNORE_INDEX = -100
ATTR_TO_SPECIAL_TOKEN = {"additional_special_tokens": ["<user>", "<assistant>", "<eot>"]}

# tokenize the example
def process(example, tokenizer):
    def tokenize(s):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s.strip()))

    user = tokenizer.user_token_id
    assistant = tokenizer.assistant_token_id
    eot = tokenizer.eot_token_id
    input_ids, labels = [], []
    
    for message in example:
        input_ids.append(user if message["role"] == "user" else assistant)
        labels.append(IGNORE_INDEX)
        content = tokenize(message["content"]) + [eot]
        input_ids.extend(content)
        labels.extend([IGNORE_INDEX]*len(content) if message["role"] == "user" else content)
            
    input_ids.append(assistant)
    labels.append(IGNORE_INDEX)
    assert len(input_ids) == len(labels)
    attention_mask = [1] * len(input_ids)
    return {'input_ids':torch.unsqueeze(torch.LongTensor(input_ids), dim=0).to("cuda")
            , 'attention_mask': torch.unsqueeze(torch.LongTensor(attention_mask), dim=0).to("cuda")}

def batch_generate(context_list, tokenizer, model, label_list):
    # Here we use traversal on only one gpu.
    results, i, choices = [], 0, ["A", "B", "C", "D"]
    choice_ids = [tokenizer.convert_tokens_to_ids(choice) for choice in choices]
    for prompt in tqdm(context_list, desc='Processing', unit='item'):
        inputs = process(prompt, tokenizer)
        label = label_list[i]
        i+=1
        with torch.no_grad():
            outputs = model(**inputs)
            last_token_logits = outputs.logits[:, -1, :]
            choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choice_logits[0])]
            results.append(pred)
        assert pred in choices
        assert label in choices
    return results

def run_multiple_dataset(dataset, output_path, tokenizer, model):
    content_list = [item["context"] for item in dataset]
    lb_list = [item['label'] for item in dataset]
    results = batch_generate(content_list, tokenizer, model, lb_list)

    formatted_results = []
    for x,y in zip(dataset, results):
        new_instance = dataset_loader.ChatGPTSchema(
            context=x['context'], metadata=x['metadata'], 
            label=x['label'], output=y, is_correct=x['label']==y)
        formatted_results.append(new_instance.to_dict())
    utils.save_jsonl(formatted_results, output_path)

    correct_numer = len([item for item in formatted_results if item["is_correct"]])
    accuracy = correct_numer / len(formatted_results)    

    return accuracy, formatted_results

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--raw_prompt_path", type=str, default="")
    parser.add_argument("--setting_name", type=str, default="z")
    parser.add_argument("--max_tokens", type=int, default=2048)
    args = parser.parse_args()

    # mkdirs
    os.makedirs(os.path.join(args.output_dir, "outputs"), exist_ok=True)

    # load tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.user_token_id, tokenizer.assistant_token_id, tokenizer.eot_token_id \
        = tokenizer.convert_tokens_to_ids(ATTR_TO_SPECIAL_TOKEN["additional_special_tokens"])
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
    model.tie_weights()
    model.eval()

    # choose datasets
    chinese_choose_datasets = ["logiqa-zh", "gaokao-chinese", "gaokao-geography", "gaokao-history","gaokao-biology", "gaokao-chemistry", "gaokao-mathqa"]
    chinese_choose_datasets.extend(["jec-qa-kd", "jec-qa-ca", "gaokao-physics"])
    setting_name = 'zero-shot' if args.setting_name=='z' else 'few-shot'

    # ***** start *****
    is_correct_all = []
    accuracy_list = []
    for dataset_name in chinese_choose_datasets:        
        dataset = dataset_loader.load_dataset(
            dataset_name, setting_name, args.dataset_dir, tokenizer=tokenizer,
            prompt_path=args.raw_prompt_path, max_tokens=args.max_tokens, verbose=False)            

        # generate and save output
        output_path = os.path.join(args.output_dir, "outputs", f'predict.{dataset_name}.{setting_name}.jsonl')
        accuracy, formatted_results = run_multiple_dataset(dataset, output_path, tokenizer, model)

        # log acc
        print(f"{dataset_name}:\t", accuracy)
        accuracy_list.append(accuracy)
        is_correct_all.extend([item["is_correct"] for item in formatted_results])

    print(f"Aeverage all examples:\t", len([item for item in is_correct_all if item])/len(is_correct_all))
    print(f"Aeverage all task:\t", np.mean(accuracy_list))


