# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import torch
import numpy as np
from math_equivalence import is_equiv

# define the datasets
single_choice_tasks = ["biology_choose","chinese_choose","history_choose","reasoning_choose","understanding_choose","ceval_choose"]
multi_choices_tasks = ['ethics_choose']
single_cloze_zh_tasks = ['chinese_cloze']
tf_tasks = ['history_tf']
math_tasks = ['math']

def extract_last_line(string):
    lines = string.split('\n')
    for item in lines[::-1]:
        if item.strip() != "":
            string = item
            break
    return string

def parse_multi_choices_tasks(string):
    string = extract_last_line(string)
    pattern = "\(*([A-F])\)*"
    match = re.findall(pattern, string)
    if match:
        return set(match)
    return set()

def parse_four_choices_candidates_tasks(answer, pred_logits, choice_parse_mode, tokenizer):
    letter_set = {"A", "B", "C", "D"}
    answer = answer.strip()
    if (len(answer)!=1 and choice_parse_mode == 'both') or choice_parse_mode == 'logits':
        pred_logits = torch.tensor(pred_logits).flatten()
        for letter in letter_set:
            assert len(tokenizer.encode(letter, add_special_tokens=False))==1
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        pred_logits[tokenizer.encode("A",add_special_tokens=False)[0]], 
                        # some tokens like '▁A' should have been taken into account when computing the logits of 'A', but we ignored it here since it will vary with tokenizer.
                        pred_logits[tokenizer.encode("B",add_special_tokens=False)[0]],
                        pred_logits[tokenizer.encode("C",add_special_tokens=False)[0]],
                        pred_logits[tokenizer.encode("D",add_special_tokens=False)[0]],
                    ]
                ),
                dim=0,
            ).detach().cpu().numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

    if choice_parse_mode=='logits':
        return pred
    elif len(answer)!=1 and choice_parse_mode=='both':
        return pred
    else:
        for c in answer:
            if c in letter_set:
                return c
        return 'E'

def parse_tf_choice(answer, pred_logits, tf_parse_mode, tokenizer):
    true_set = ['正确','对','T','√','V']
    false_set = ['错误','错','F','×','X']

    answer = answer.strip()
    true_set_search = re.findall(r'(?:{})'.format('|'.join(true_set)),answer)
    false_set_search = re.findall(r'(?:{})'.format('|'.join(false_set)),answer)

    if tf_parse_mode in ['both','logits']:
        pred_logits = torch.tensor(pred_logits)
        pred_logits = pred_logits.flatten()
        ture_logits = float(0)
        false_logits = float(0)
        # detect if there is a special start token for chinese token
        token_list = []
        for test_text in ['正确','错误','正','确','错','误']:
            token_list.append(tokenizer.encode(test_text,add_special_tokens=False)[0])
        if len(token_list)==6 and len(set(token_list))==1:
            start_token_zh = token_list[0]
        
        # calculate the scores
        for t,f in zip(true_set, false_set):
            tokenize_t = tokenizer.encode(t)
            tokenize_f = tokenizer.encode(f)           
            if len(tokenize_t)==1 and len(tokenize_f)==1:
                ture_logits+=pred_logits[tokenize_t[0]]
                false_logits+=pred_logits[tokenize_f[0]]
            elif len(tokenize_t)==2 and len(tokenize_f)==2 and tokenize_t[0]==start_token_zh and tokenize_f[0]==start_token_zh:
                ture_logits+=pred_logits[tokenize_t[1]]
                false_logits+=pred_logits[tokenize_f[1]]               
        
        probs = (torch.nn.functional.softmax(torch.tensor([ture_logits,false_logits]),dim=0,).detach().cpu().numpy())
        pred = {0: "T", 1: "F"}[np.argmax(probs)]

    # We prioritize the results of regularization
    if len(true_set_search)>0 and len(false_set_search)==0:
        answer = 'T'
    elif len(false_set_search)>0 and len(true_set_search)==0:
        answer = 'F'
    elif tf_parse_mode in ['both','logits']:
        answer = pred
    else:
        answer = 'E'
        
    return answer

def parse_single_cloze_zh(answer):
    answer = answer.strip()
    single_cloze_match = re.findall(r"[\u4e00-\u9fff]+.*?[^\w\s]*",answer) # here we extract the chinese string between punc
    if len(single_cloze_match)>0:
        return (single_cloze_match[0]).strip()
    else:
        return answer

def parse_math_tasks(raw_string):
    # If the "raw_string" is "\\boxed{x + y = z}", it will be processed into "z"
    def remove_boxed(s):
        if "\\fbox" in s:
            left = "\\boxed{"
        else:
            left = "\\boxed{"
        try:
            assert s[:len(left)] == left
            assert s[-1] == "}"
            answer = s[len(left):-1]
            if "=" in answer:
                answer = answer.split("=")[-1].lstrip(" ")
            return answer
        except:
            return None

    # If there are many "\\boxed{" in the "string", only the last one will be extracted (the use of "rfind").
    # If "\\boxed{" can't be found, then try to find "\\fbox"
    # The number of "{" and "}" must match with each other
    def last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx == None:
            retval = None
        else:
            retval = string[idx:right_brace_idx + 1]

        return retval

    #  Find the answer in the last part whose pattern is "\$(.*)\$", only the last one will be extracted
    def get_answer_with_dollar_sign(s):
        first_pattern = "\$(.*)\$"
        last_match = None
        matches = re.findall(first_pattern, s)
        if matches:
            last_match = matches[-1]
            if "=" in last_match:
                last_match = last_match.split("=")[-1].lstrip(" ")
        return last_match

    # If we can't find "\\boxed" or a pattern like "\$(.*)\$", we will firstly look for "=" directly,
    # If we still can't find "=", we will just find a number which starts with "$"("$" won't be included in the final result)
    # and can have fraction, but it can't be followed with some other text. Only the last one will be extracted
    def get_answer_without_dollar_sign(s):
        last_match = None
        if "=" in s:
            last_match = s.split("=")[-1].lstrip(" ").rstrip(".")
            if "\\n" in last_match:
                last_match = last_match.split("\\n")[0]        
        else:
            pattern = "(?:\\$)?(?:-)?\d+(?:\.\d+)?(?![\w\d])"
            matches = re.findall(pattern, s)
            if matches:
                last_match = matches[-1]
                last_match = last_match.strip('$')
        return last_match

    # Some special case for our dataset is considered here
    def get_answer_special_case(s):
        s = s.strip()
        s_ori = s
        if len(s.split("/")) == 2:
            if bool(re.search(r"[^a-zA-Z0-9]", s.split("/")[0])) or bool(re.search(r"[^a-zA-Z0-9]", s.split("/")[1])):
                return None
            else:
                return s
        s = s.replace('，',',')
        if re.match(r'^[,-?\d]+$',s) is not None:
            return s
        s = s.replace(' ','')
        find_sqrt = re.findall(r'(sqrt\([^)]*\))',s)
        if len(find_sqrt)>0:
            return find_sqrt[-1]
        find_bracket_contents = re.findall(r'(\[.*?\]|\(.*?\))',s)
        if len(find_bracket_contents)>0:
            return find_bracket_contents[-1] 
        return None
    
    def poset_process_special_case(s):
        # For the situation: {'answer': '\\frac{3}{4}, -\\frac{3}{4}', 'output': '答案\n$\\frac{3}{4}$或$-\\frac{3}{4}$', 'parse': '\\frac{3}{4}$或$-\\frac{3}{4}'}
        if '或' in s:
            s = s.replace('或',', ')
        if '$' in s:
            s = s.replace('$','')
        return s       
        
    if "答案" in raw_string:
        raw_string = (raw_string.split('答案')[-1]).strip()

    # If "\\boxed" in raw_string and "\\fbox" not in raw_string, last_boxed will be None
    last_boxed = last_boxed_only_string(raw_string)
    
    if last_boxed is not None:
        answer = remove_boxed(last_boxed)
    else:
        answer = get_answer_with_dollar_sign(raw_string)
        if not answer:
            answer = get_answer_special_case(raw_string)
        if not answer:
            answer = get_answer_without_dollar_sign(raw_string)
    if answer is not None:
        answer = poset_process_special_case(answer)
    return answer

# Parse the output
def post_process_single_sample(prediction, dataset_name, pred_logits, choice_parse_mode, tokenizer, tf_parse_mode='text'):
    if dataset_name in single_choice_tasks:
        answer = parse_four_choices_candidates_tasks(prediction, pred_logits, choice_parse_mode, tokenizer)
    
    # we identify the t/f result based on output
    if dataset_name in tf_tasks:
        answer = parse_tf_choice(prediction, pred_logits, tf_parse_mode, tokenizer)
    
    # we process chinese single-cloze examples here
    if dataset_name in single_cloze_zh_tasks:
        answer = parse_single_cloze_zh(prediction)

    if dataset_name in multi_choices_tasks:
        answer = parse_multi_choices_tasks(prediction)
    
    if dataset_name in math_tasks:
        answer = parse_math_tasks(prediction)
  
    return answer

def evaluate_single_sample(dataset_name, prediction, label, label_content=None, verbose=False):
    if dataset_name in math_tasks:
        return is_equiv(prediction, label, verbose)
    if label_content is None:
        return prediction == label
    else:
        return prediction == label or prediction.strip()==label_content.strip()  