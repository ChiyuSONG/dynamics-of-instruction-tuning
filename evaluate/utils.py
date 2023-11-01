import torch
import parse_utils
import re
import math

class ScoreSchema(object):
    def __init__(self, score=0, score_expected=0):
        self.score = score
        self.score_expected = score_expected
        
    def to_dict(self):
        return {
            "score": self.score,
            "score_expected": self.score_expected
        }

def chinese_score(sample, tokenizer=None, choice_parse_mode='both'):
    # 4-choices-candidates tasks and single-blank filling tasks
    assert len(sample['messages'])==2 and sample['type']=="chinese" and sample['question_format']==0
    model_output = sample['generated_response']
    label = sample['messages'][-1]['content']
    if label in ['A','B','C','D']:
        dataset_name = "chinese_choose"
        pred_logits = sample['generated_score']
    else:
        dataset_name = "chinese_cloze"
        pred_logits = None
    parse_result = parse_utils.post_process_single_sample(model_output, dataset_name, pred_logits, choice_parse_mode, tokenizer)
    is_correct = parse_utils.evaluate_single_sample(dataset_name, parse_result, label)
    # print({"model_output":model_output,"parse_result":parse_result,"label":label})
 
    score = 1 if is_correct else 0
    score_expected = 0.25 if dataset_name == "chinese_choose" else 0
    return ScoreSchema(score, score_expected).to_dict()



def biology_score(sample, tokenizer=None, choice_parse_mode='both'): # available: ['both','logits','text']
    # 4-choices-candidates tasks
    assert len(sample['messages'])==2 and sample['type']=="biology" and sample['question_format']==0
    model_output = sample['generated_response']
    label = sample['messages'][-1]['content']
    assert label in ['A','B','C','D']
    dataset_name = "biology_choose"
    pred_logits = sample['generated_score']

    parse_result = parse_utils.post_process_single_sample(model_output, dataset_name, pred_logits, choice_parse_mode, tokenizer)
    is_correct = parse_utils.evaluate_single_sample(dataset_name, parse_result, label)
    # print({"model_output":model_output,"parse_result":parse_result,"label":label})

    score = 1 if is_correct else 0
    score_expected = 0.25
    return ScoreSchema(score, score_expected).to_dict()



def history_score(sample, tokenizer=None, choice_parse_mode='both', tf_parse_mode='text'): # available: ['both','logits','text']
    # 4-choices-candidates tasks and ture/false tasks
    assert len(sample['messages'])==2 and sample['type']=="history" and sample['question_format']==0
    model_output = sample['generated_response']
    label = sample['messages'][-1]['content']
    if label in ['A','B','C','D']:
        dataset_name = "history_choose"
        pred_logits = sample['generated_score']
    else:
        dataset_name = "history_tf"
        pred_logits = sample['generated_score']
        choice_parse_mode = None
        assert label in ['正确','对','T','√','V','错误','错','F','×','X']
        if label in ['正确','对','T','√','V']:
            label = 'T'
        else:
            label = 'F'
    parse_result = parse_utils.post_process_single_sample(model_output, dataset_name, pred_logits, choice_parse_mode, tokenizer, tf_parse_mode)
    is_correct = parse_utils.evaluate_single_sample(dataset_name, parse_result, label)
    # print({"model_output":model_output,"parse_result":parse_result,"label":label})

    score = 1 if is_correct else 0
    score_expected = 0.25 if dataset_name == "history_choose" else 0.5
    return ScoreSchema(score, score_expected).to_dict()



def reasoning_score(sample, tokenizer=None, choice_parse_mode='both'): # available: ['both','logits','text']
    # 4-choices-candidates tasks
    assert len(sample['messages'])==2 and sample['type']=="reasoning" and sample['question_format']==0
    model_output = sample['generated_response']
    label = sample['messages'][-1]['content']
    assert label in ['A','B','C','D']
    dataset_name = "reasoning_choose"
    pred_logits = sample['generated_score']

    parse_result = parse_utils.post_process_single_sample(model_output, dataset_name, pred_logits, choice_parse_mode, tokenizer)
    is_correct = parse_utils.evaluate_single_sample(dataset_name, parse_result, label)
    # print({"model_output":model_output,"parse_result":parse_result,"label":label})

    score = 1 if is_correct else 0
    score_expected = 0.25
    return ScoreSchema(score, score_expected).to_dict()



def understanding_score(sample, tokenizer=None, choice_parse_mode='both'): # available: ['both','logits','text']
    # multi-turn dialogue, including 3-choices-candidates tasks and 4-choices-candidates tasks
    assert len(sample['messages'])>1 and sample['type']=="understanding" and sample['question_format']==0
    model_output = sample['generated_response']
    label = sample['messages'][-1]['content']
    assert label in ['A','B','C','D'] and sample['messages'][-1]['role']=='assistant'
    assert re.search(r'\nA\..*\nB\..*\nC\.',sample['messages'][-2]['content'])
    dataset_name = "understanding_choose"
    pred_logits = sample['generated_score']

    parse_result = parse_utils.post_process_single_sample(model_output, dataset_name, pred_logits, choice_parse_mode, tokenizer)
    is_correct = parse_utils.evaluate_single_sample(dataset_name, parse_result, label)
    # print({"model_output":model_output,"parse_result":parse_result,"label":label})

    score = 1 if is_correct else 0
    score_expected = 0.25 if re.search(r'\nA\..*\nB\..*\nC\..*\nD\.',sample['messages'][-2]['content']) else 0.33
    return ScoreSchema(score, score_expected).to_dict()



def ethics_score(sample, tokenizer=None, choice_parse_mode='both'):
    # multi-choices tasks
    assert len(sample['messages'])==2 and sample['type']=="ethics" and sample['question_format']==0
    model_output = sample['generated_response']
    label = sample['messages'][-1]['content']
    label_ori = label
    label = parse_utils.parse_multi_choices_tasks(label)
    dataset_name = "ethics_choose"

    parse_result = parse_utils.post_process_single_sample(model_output, dataset_name, pred_logits=None, choice_parse_mode=None, tokenizer=None)
    is_correct = parse_utils.evaluate_single_sample(dataset_name, parse_result, label)
    # print({"model_output":model_output,"parse_result":parse_result,"label":label,"label_ori":label_ori})

    score = 1 if is_correct else 0
    score_expected = 1/(math.comb(4, len(label)))
    return ScoreSchema(score, score_expected).to_dict()



def math_score(sample, tokenizer=None, choice_parse_mode='both'):
    # multi-choices tasks
    assert len(sample['messages'])==2 and sample['type']=="math" and sample['question_format']==0
    model_output = sample['generated_response']
    label = sample['answer']
    dataset_name = "math"

    parse_result = parse_utils.post_process_single_sample(model_output, dataset_name, pred_logits=None, choice_parse_mode=None, tokenizer=None)
    is_correct = parse_utils.evaluate_single_sample(dataset_name, parse_result, label)
    # print({"model_output":model_output,"parse_result":parse_result,"label":label,"label_ori":label})
    
    score = 1 if is_correct else 0
    score_expected = 0
    return ScoreSchema(score, score_expected).to_dict()



def ceval_physician_fshot_score(sample, tokenizer=None, choice_parse_mode='both'): # available: ['both','logits','text']
    # 4-choices-candidates tasks using few-shot(5-shot) in prompt
    assert len(sample['messages'])==12 and sample['type']=="ceval_physician_fshot" and sample['question_format']==0
    model_output = sample['generated_response']
    label = sample['messages'][-1]['content']
    assert label in ['A','B','C','D']
    dataset_name = "ceval_choose"
    pred_logits = sample['generated_score']

    parse_result = parse_utils.post_process_single_sample(model_output, dataset_name, pred_logits, choice_parse_mode, tokenizer)
    is_correct = parse_utils.evaluate_single_sample(dataset_name, parse_result, label)
    # print({"model_output":model_output,"parse_result":parse_result,"label":label})

    score = 1 if is_correct else 0
    score_expected = 0.25
    return ScoreSchema(score, score_expected).to_dict()



def ceval_urban_and_rural_planner_fshot_score(sample, tokenizer=None, choice_parse_mode='both'): # available: ['both','logits','text']
    # 4-choices-candidates tasks using few-shot(5-shot) in prompt
    assert len(sample['messages'])==12 and sample['type']=="ceval_urban_and_rural_planner_fshot" and sample['question_format']==0
    model_output = sample['generated_response']
    label = sample['messages'][-1]['content']
    assert label in ['A','B','C','D']
    dataset_name = "ceval_choose"
    pred_logits = sample['generated_score']

    parse_result = parse_utils.post_process_single_sample(model_output, dataset_name, pred_logits, choice_parse_mode, tokenizer)
    is_correct = parse_utils.evaluate_single_sample(dataset_name, parse_result, label)
    # print({"model_output":model_output,"parse_result":parse_result,"label":label})

    score = 1 if is_correct else 0
    score_expected = 0.25
    return ScoreSchema(score, score_expected).to_dict()



def ceval_teacher_qualification_fshot_score(sample, tokenizer=None, choice_parse_mode='both'): # available: ['both','logits','text']
    # 4-choices-candidates tasks using few-shot(5-shot) in prompt
    assert len(sample['messages'])==12 and sample['type']=="ceval_teacher_qualification_fshot" and sample['question_format']==0
    model_output = sample['generated_response']
    label = sample['messages'][-1]['content']
    assert label in ['A','B','C','D']
    dataset_name = "ceval_choose"
    pred_logits = sample['generated_score']

    parse_result = parse_utils.post_process_single_sample(model_output, dataset_name, pred_logits, choice_parse_mode, tokenizer)
    is_correct = parse_utils.evaluate_single_sample(dataset_name, parse_result, label)
    # print({"model_output":model_output,"parse_result":parse_result,"label":label})

    score = 1 if is_correct else 0
    score_expected = 0.25
    return ScoreSchema(score, score_expected).to_dict()
