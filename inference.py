import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    set_seed,
    GenerationConfig
)
from train_sft import IGNORE_INDEX, DataCollatorForSupervisedDataset, ATTR_TO_SPECIAL_TOKEN



def process(batch, tokenizer):
    processed = []
    user = tokenizer.user_token_id
    assistant = tokenizer.assistant_token_id
    eot = tokenizer.eot_token_id

    def tokenize(s):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s.strip()))

    for example in batch:
        input_ids = []
        labels = []
        messages = []
        messages.extend(example["messages"])
        for message in messages:
            input_ids.append(user if message["role"] == "user" else assistant)
            labels.append(IGNORE_INDEX)
            content = tokenize(message["content"]) + [eot]
            input_ids.extend(content)
            labels.extend([IGNORE_INDEX]*len(content) if message["role"] == "user" else content)
        input_ids.append(assistant)
        labels.append(IGNORE_INDEX)
        assert len(input_ids) == len(labels)
        attention_mask = [1] * len(input_ids)
        processed.append( {'input_ids':torch.LongTensor(input_ids), 'labels': torch.LongTensor(labels), 'attention_mask': torch.LongTensor(attention_mask)} )

    return processed


class Assistant:
    def __init__(self, model_name_or_path):
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        tokenizer.padding_side = "left"
        tokenizer.user_token_id, tokenizer.assistant_token_id, tokenizer.eot_token_id \
            = tokenizer.convert_tokens_to_ids(ATTR_TO_SPECIAL_TOKEN["additional_special_tokens"])
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
        model.tie_weights()
        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        self.seed = 0
        # use greedy decoding as default
        self.config = GenerationConfig(
            max_new_tokens=1024,
            min_length=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id,
                          tokenizer.eot_token_id, tokenizer.user_token_id, tokenizer.assistant_token_id],
        )
        set_seed(self.seed)

    def inference(self, batch):
        processed = process(batch, tokenizer=self.tokenizer)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer, pad_to_multiple_of=8)
        inputs = data_collator(processed)
        for key in inputs:
            inputs[key].to("cuda")
        outputs = self.model.generate(
            **inputs,
            generation_config = self.config
        )
        scores = outputs.scores[-1]
        sequences = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        prefix = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        responses = [sequences[i][len(prefix[i]) : ].strip() for i in range(len(sequences))]
        return responses, scores
