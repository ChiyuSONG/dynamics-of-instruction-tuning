import os
import sys
from pathlib import Path
from functools import partial
import math
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
import logging
import torch
import transformers
import datasets
from datasets import load_dataset, concatenate_datasets, disable_caching
disable_caching()
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed
)
from transformers import get_polynomial_decay_schedule_with_warmup
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)
IGNORE_INDEX = -100
ATTR_TO_SPECIAL_TOKEN = {"additional_special_tokens": ["<user>", "<assistant>", "<eot>"]}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    llama: bool = field(
        default=True,
        metadata={"help": "use Llama model"}
    )


@dataclass
class DataArguments:
    train_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory of train files."}
    )
    valid_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory of valid files."},
    )
    use_data_cache: bool = field(
        default=False,
        metadata={"help": "Whether to store and load cached datasets."},
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training or not."},
    )
    end_learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The end learning rate for setting LR scheduler."}
    )
    resume_training: bool = field(
        default=False,
        metadata={"help": "Whether to resume training from the checkpoint in output_dir"},
    )
    max_seq_len: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length."},
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "enable gradient_checkpointing"}
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "enable low_cpu_mem_usage"}
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "ddp_find_unused_parameters"}
    )


def process(examples, tokenizer, max_seq_len):
    input_ids = []
    labels = []
    attention_mask = []
    user = tokenizer.user_token_id
    assistant = tokenizer.assistant_token_id
    eot = tokenizer.eot_token_id

    def tokenize(s):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s.strip()))

    for messages in examples["messages"]:
        input_id = []
        label = []
        for message in messages:
            input_id.append(user if message["role"] == "user" else assistant)
            label.append(IGNORE_INDEX)
            content = tokenize(message["content"]) + [eot]
            input_id.extend(content)
            label.extend([IGNORE_INDEX]*len(content) if message["role"] == "user" else content)
        assert len(input_id) == len(label)
        if len(input_id) > max_seq_len:
            input_id = input_id[:max_seq_len]
            label = label[:max_seq_len]
            logger.warning(f"Found overlong training data, length: {len(input_id)}. Trimmed.")
        input_ids.append(input_id)
        labels.append(label)
        attention_mask.append([1] * len(input_id))

    return {'input_ids':input_ids, 'labels': labels, 'attention_mask': attention_mask}


def buid_instruction_dataset(data_dir, tokenizer, use_data_cache, max_seq_len):
    merged_dataset = []
    path = Path(data_dir)
    data_files = [os.path.join(path, file.name) for file in path.glob("*.json")]
    for data_file in data_files:
        cache_path = str(os.path.dirname(data_file))
        cache_path = os.path.join(cache_path, "cache_"+os.path.basename(data_file).split('.')[0])

        if use_data_cache and os.path.exists(cache_path):
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'training datasets-{data_file} has been loaded from disk.')
        else:
            raw_dataset = load_dataset("json", data_files=data_file, cache_dir=cache_path)
            processed_dataset = raw_dataset.map(
                partial(process, tokenizer=tokenizer, max_seq_len=max_seq_len),
                batched=True,
                remove_columns=list(raw_dataset["train"].features),
                keep_in_memory=False,
                desc="processing on dataset",
            )
            os.makedirs(cache_path, exist_ok=True)
            processed_dataset.save_to_disk(cache_path)
            logger.info(f'Cached processed dataset at {cache_path}.')

        processed_dataset.set_format('torch')
        merged_dataset.append(processed_dataset['train'])
    merged_dataset = concatenate_datasets(merged_dataset)
    return merged_dataset


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))

        max_label_length = max(len(l) for l in labels)
        if self.pad_to_multiple_of is not None:
            max_label_length = (
                (max_label_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        input_ids = self.pad_sequence(input_ids, self.tokenizer.pad_token_id, max_label_length)
        labels = self.pad_sequence(labels, IGNORE_INDEX, max_label_length)
        attention_mask = self.pad_sequence(attention_mask, 0, max_label_length)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

    def pad_sequence(self, feature, padding_value, max_label_length):
        for idx, instance in enumerate(feature):
            remainder = torch.LongTensor( [padding_value] * (max_label_length - len(instance)) )
            feature[idx] = torch.cat((instance, remainder), 0) if self.tokenizer.padding_side == "right" \
                else torch.cat((remainder, instance), 0)
        return torch.stack(feature, dim = 0)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if training_args.local_rank <= 0 else logging.WARN,
                        handlers=[logging.StreamHandler(sys.stdout)]
                        )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"Distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


    # Load last checkpoint if resume otherwise start from scratch.
    last_checkpoint = None
    if training_args.resume_training:
        if os.path.isdir(training_args.output_dir):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) has no valid checkpoint. "
                )
        else:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) doesn't exist and cannot resume checkpoint from it."
            )
    else:
        if os.path.isdir(training_args.output_dir) and len(os.listdir(training_args.output_dir)) > 0 and not training_args.overwrite_output_dir:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if model_args.llama:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)
        logger.info("Set the eos_token_id and bos_token_id of LLama model tokenizer")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    tokenizer.padding_side = "left"

    ATTR_TO_SPECIAL_TOKEN["additional_special_tokens"] += tokenizer.additional_special_tokens
    if tokenizer.pad_token_id is None:
        ATTR_TO_SPECIAL_TOKEN["pad_token"] = "<pad>"
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    logger.info("Added {} more special tokens".format(num_added_tokens))
    logger.info("<pad> token id is {}".format(tokenizer.pad_token_id))
    tokenizer.user_token_id, tokenizer.assistant_token_id, tokenizer.eot_token_id \
        = tokenizer.convert_tokens_to_ids(ATTR_TO_SPECIAL_TOKEN["additional_special_tokens"])
    logger.info("<user> token id is {}".format(tokenizer.user_token_id))
    logger.info("<assistant> token id is {}".format(tokenizer.assistant_token_id))
    logger.info("<eot> token id is {}".format(tokenizer.eot_token_id))


    logger.info("building dataset...")
    train_dataset = buid_instruction_dataset(data_args.train_dir, tokenizer, data_args.use_data_cache, training_args.max_seq_len)
    valid_dataset = buid_instruction_dataset(data_args.valid_dir, tokenizer, data_args.use_data_cache, training_args.max_seq_len)
    logger.info(f"Num train_samples: {len(train_dataset)}")
    logger.info(f"Num eval_samples: {len(valid_dataset)}")
    logger.info("example data:\n")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, pad_to_multiple_of=8)
    example_data = data_collator(train_dataset.select(range(10)))
    logger.info("***** example of processed data below *****")
    for i in range(1):
        example_input_id = example_data['input_ids'][i].tolist()
        example_label = example_data['labels'][i].tolist()
        example_attention_mask = example_data['attention_mask'][i].tolist()
        logger.info(tokenizer.decode(example_input_id))
        logger.info(example_input_id)
        logger.info(example_label)
        logger.info(example_attention_mask)
    logger.info("***** example of processed data above *****")

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch_dtype, low_cpu_mem_usage=training_args.low_cpu_mem_usage)
    model.resize_token_embeddings(len(tokenizer))

    class CustomTrainer(Trainer):
        def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
            """
            Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
            passed as an argument.

            Args:
                num_training_steps (int): The number of training steps to do.
            """
            if self.lr_scheduler is None:
                self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    lr_end=training_args.end_learning_rate
                )
            return self.lr_scheduler

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    real_bs = training_args.per_device_train_batch_size * training_args.world_size * training_args.gradient_accumulation_steps
    total_steps = math.ceil(len(train_dataset) / real_bs) * training_args.num_train_epochs

    logger.info(f"  world_size = {training_args.world_size}")
    logger.info(f"  Real train batch size = {real_bs}")
    logger.info(f"  Total train steps = {total_steps}")
    logger.info(f"Using {training_args.half_precision_backend} half precision backend.")
    logger.info("***** Start Training *****")

    if not training_args.resume_training:
        assert last_checkpoint is None

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    # training summary
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    try:
        perplexity = math.exp(metrics["train_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["train_ppl"] = perplexity
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # valid summary
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(valid_dataset)
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["eval_ppl"] = perplexity
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
