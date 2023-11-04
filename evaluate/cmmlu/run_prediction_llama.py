import torch
import numpy as np
import argparse
from src.mp_utils import choices, format_example, gen_prompt, run_eval

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
IGNORE_INDEX = -100
ATTR_TO_SPECIAL_TOKEN = {"additional_special_tokens": ["<user>", "<assistant>", "<eot>"]}

def softmax(x):
    exp_z = np.exp(x - max(x))
    prob = exp_z/np.sum(exp_z)
    return prob

def process(example, tokenizer):
    user = tokenizer.user_token_id
    assistant = tokenizer.assistant_token_id
    eot = tokenizer.eot_token_id

    def tokenize(s):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s.strip()))

    input_ids, labels = [], []
    input_ids.append(user)
    labels.append(IGNORE_INDEX)
    content = tokenize(example) + [eot]
    input_ids.extend(content)
    labels.extend([IGNORE_INDEX]*len(content))
    input_ids.append(assistant)
    labels.append(IGNORE_INDEX)
    
    assert len(input_ids) == len(labels)
    attention_mask = [1] * len(input_ids)
    
    return {'input_ids':torch.unsqueeze(torch.LongTensor(input_ids), dim=0).to("cuda")
            , 'attention_mask': torch.unsqueeze(torch.LongTensor(attention_mask), dim=0).to("cuda")}


def eval_new(model, tokenizer, subject, dev_df, test_df, num_few_shot, max_length):
    choice_ids = [tokenizer.convert_tokens_to_ids(choice) for choice in choices]
    cors, all_conf, all_preds = [], [], []

    for i in tqdm(range(test_df.shape[0])):
        prompt_end = format_example(test_df, i, include_answer=False)

        prompt = gen_prompt(dev_df=dev_df,
                            subject=subject,
                            prompt_end=prompt_end,
                            num_few_shot=num_few_shot,
                            tokenizer=tokenizer,
                            max_length=max_length)

        inputs = process(prompt, tokenizer)

        label = test_df.iloc[i, test_df.shape[1] - 1]
        with torch.no_grad():
            outputs = model(**inputs)
            last_token_logits = outputs.logits[:, -1, :]
            choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
            conf = softmax(choice_logits[0])[choices.index(label)]
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choice_logits[0])]

        all_preds += pred
        all_conf.append(conf)
        cors.append(pred == label)

    acc = np.mean(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return acc, all_preds, all_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--load_in_8bit", action='store_true')
    parser.add_argument("--with_conf", action='store_true')
    parser.add_argument("--cot", action='store_true')
    parser.add_argument("--sample_ratio", type=float, default=1)
    args = parser.parse_args()

    print('*****'+'info'+"*****")
    print(f"sample_ratio:{args.sample_ratio}")
    print(f"save_dir:{args.save_dir}")
    print(f"data_dir:{args.data_dir}")
    print(f"model_name_or_path:{args.model_name_or_path}")

    # TODO: better handle
    tokenizer_class = LlamaTokenizer if 'llama' in args.model_name_or_path else AutoTokenizer
    model_class = LlamaForCausalLM if 'llama' in args.model_name_or_path else AutoModelForCausalLM
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        trust_remote_code=True,
                                        load_in_8bit=args.load_in_8bit,
                                        device_map="auto"
                                        )
    tokenizer.user_token_id, tokenizer.assistant_token_id, tokenizer.eot_token_id \
        = tokenizer.convert_tokens_to_ids(ATTR_TO_SPECIAL_TOKEN["additional_special_tokens"])

    with torch.no_grad():
        run_eval(model, tokenizer, eval_new, args)