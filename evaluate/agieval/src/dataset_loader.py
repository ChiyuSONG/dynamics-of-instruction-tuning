# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pandas as pd
import os
import ast
from src.utils import read_jsonl


class ChatGPTSchema(object):
    def __init__(self, context=None, metadata="", label="", output="", is_correct=""):
        self.context = context
        self.metadata = metadata
        self.label = label
        self.output = output
        self.is_correct = is_correct

    def to_dict(self):
        return {
            "context": self.context,
            "metadata": self.metadata,
            "label": self.label,
            "output": self.output,
            "is_correct": self.is_correct
        }


# Here we replace "A: The answer is" with "A: " for cloze tasks
def convert_zero_shot(line):
    passage = line["passage"] if line["passage"] is not None else ""

    question_input= passage + "问题：" + line["question"] + " " \
        + "选项：" + " ".join(line["options"]) + "\n" + \
        "答案：从A到D, 我们应选择"
                                            
    return [
        {"role": "user", "content": question_input},
        ]

# The prefix for the final anwser(["The answer is therefore", "答案是"]) 
# is added for few-shot setting
def convert_few_shot(line, demo, n_shot):
    passage = line["passage"] if line["passage"] is not None else ""
    question = line["question"]
    options = line["options"] if line["options"] is not None else ""

    question_input = "问题 {}.   ".format(n_shot + 1) + passage + " " + question + "\n" \
        + "从以下选项中选择:    " + " ".join(options) + "\n" \
        + "答案：从A到D, 我们应选择"

    return demo + [
        {"role": "user", "content": question_input},
    ]


# process few-shot raw_prompts
# We replace "答案是 " with "答案：从A到{}, 我们应选择" for chinese dataset
# !!!!! For multi-choices tasks, we remove the example with multi-choice here
def combine_prompt(prompt_path, dataset_name):
    demostrations, contexts, context_row = [], [], [0, 1, 3, 5, 7, 9]
    raw_prompts_context = pd.read_csv(prompt_path, header=0, skiprows=lambda x: x not in context_row,
                                      keep_default_na=False)
    for line in list(raw_prompts_context[dataset_name]):
        if line:
            contexts.append(ast.literal_eval(line))

    print(5*'\n')
    for idx, con in enumerate(contexts):
        if len(con["label"])>1:
            continue
        elif isinstance(con["label"],list):
            con["label"] = con["label"][0]

        passage = con["passage"] if con["passage"] is not None else ""
        question = con["question"]
        options = con["options"] if con["options"] is not None else ""
        label = con["label"] if con["label"] is not None else ""

        try:
            option_string = "ABCDEFG"
            count = len(options)
            if count == 1:
                count = 4
            question_input = "问题 {}.   ".format(idx + 1) + passage + " " + question + "\n" \
                            + "从以下选项中选择:    " + " ".join(options) + "\n" \
                            + "答案：从A到{}, 我们应选择".format(option_string[count - 1])
            question_output = "{}".format(label)
        except:
            raise ValueError(f"During loading few-sot examples, found unknown dataset: {dataset_name}")
        demostrations.append((question_input, question_output))

    return demostrations

def concat_prompt(demos, max_tokens, tokenizer, verbose=True, prompt_max_length=0):
    answers, sentences = [], ""
    max_tokens = max_tokens-48-prompt_max_length

    for i in range(len(demos)):
        answers += [
            {"role": "user", "content": demos[i][0]},
            {"role": "assistant", "content": demos[i][1]},
        ]
        sentences += answers[-2]['content']
        sentences += answers[-1]['content']

        if len(tokenizer.encode(sentences)) > max_tokens:
            answers.pop()
            answers.pop()
            break
    if verbose:
        print("max_token is set as {} but the actual_tokens is {}, and there are {} shots finally."
              .format(max_tokens, len(tokenizer.encode(sentences)), len(answers)//2))
    return answers, len(answers)//2

def load_dataset(dataset_name, setting_name, parent_path, tokenizer=None,
                 prompt_path=None, max_tokens=None, verbose=False):
    # load data
    data_path = os.path.join(parent_path, dataset_name + ".jsonl")
    loaded_jsonl = read_jsonl(data_path)

    # "prompt_max_length" will be used to calculate 
    # how many demos can be used so that the examples won't exceed the max_tokens
    prompt_max_length = {}
    for meta_idx, line in enumerate(loaded_jsonl):
        passage = line["passage"] if line["passage"] is not None else ""
        question = line["question"] if line["question"] is not None else ""
        options = line["options"] if line["options"] is not None else ""
        options = " ".join(options)
        prompt_length = len(tokenizer.encode(''.join([passage, question, options])))
        prompt_max_length[meta_idx]=prompt_length

    # load few-shot prompt
    if setting_name == "few-shot":
        processed_demos = combine_prompt(prompt_path, dataset_name)

    processed = []
    for meta_idx, line in enumerate(loaded_jsonl):
        # we include some multi-choce datasets, here, 
        # only those examples with 4 choices are loaded
        if len(line["label"])!=1:
            continue
        elif isinstance(line["label"],list):
            line["label"] = line["label"][0]
        assert len(line['options'])==4
        assert line['label'] in ['A','B','C','D']

        # load few-shot examples
        if setting_name == "few-shot":
            chosen_prompt, n_shot = concat_prompt(
                processed_demos, max_tokens, tokenizer, 
                verbose=verbose, prompt_max_length=prompt_max_length[meta_idx])

        # process data
        if setting_name == "zero-shot":
            ctxt = convert_zero_shot(line)
        elif setting_name == "few-shot":
            ctxt = convert_few_shot(line, chosen_prompt, n_shot)

        # re-format data
        try:
            new_instance = ChatGPTSchema(context=ctxt, metadata=meta_idx, label=line['label'])
            processed.append(new_instance.to_dict())
        except NameError:
            print("Dataset not defined.")
    return processed



if __name__ == "__main__":
    print('start')
