# Dynamics of Instruction Tuning: Each Ability of Large Language Models Has Its Own Growth Pace

This [paper](https://arxiv.org/abs/2310.19651) investigated how the underlying abilities of Large Language Models (LLMs), such as creative writing, code generation, and logical reasoning, develop at varying paces during instruction tuning. We systematically studied the effects of data volume, parameter size (7b-33b), and data construction methods on the growth of each ability.

- The codebase and commands are provided to reproduce our experimental results.
- The human-curated dataset for training and evaluation can be found [here](https://huggingface.co/datasets/ChiyuSONG/dynamics-of-instruction-tuning).
- We further validate the efficacy of our data construction on other foundation models such as Baichuan2. The deployable model checkpoints can be found in this [repo](https://github.com/ChiyuSONG/data-efficient-training-of-LLMs). :grin:**Have Fun!**

<p align="center" width="100%">
      <img src="img/way_to_agi.jpg" alt="Each ability of LLMs has its own growth pace during instruction tuning." style="width: 56%; min-width: 200px; display: block; margin: auto;">
</p>

## Dependencies
The code is implemented using python 3.9 and PyTorch v2.0.1
```bash
pip install -r requirements.txt
```

## Training
1. Get the dataset and move its content to "data/":
```bash
git clone https://huggingface.co/datasets/ChiyuSONG/dynamics-of-instruction-tuning
```

2. Get the foundation models from [here](https://github.com/ymcui/Chinese-LLaMA-Alpaca). They are LLaMA series models (Touvron et al., 2023) with further pre-training in Chinese. We use the "Plus" version that ranges from 7b to 33b in our experiments.


3. To train models under different experimental settings:
```bash
# choices for data_type:
# ["curated-10", "curated-40", "curated-160", "curated-640", "curated-2560", "curated-10000","synthetic-10", "synthetic-40", "synthetic-160", "synthetic-640", "synthetic-2560", "synthetic-10000","synthetic-40960", "baseline", "reconstruct", "maximum", "mix-0", "mix-2560", "mix-40960"]
bash run_train.sh --data_type **the_setting_you_chose** --model_size 7b --model_name_or_path **path_to_foundation_model** --batch_size 8 --gradient_accumulation 1
```
&nbsp;&nbsp;&nbsp;&nbsp;Training logs and model checkpoints will be saved in "/runs".

## Evaluation
### Evaluate models on human-curated valid/test sets:

1. Generate predictions on the valid/test questions.
```bash
export CUDA_VISIBLE_DEVICES=0
time python -u evaluate/pred.py --model_name_or_path **path_to_saved_checkpoint** --eval_data_path data/curated/valid #or test
```
&nbsp;&nbsp;&nbsp;&nbsp;The generated answers will be saved in "evaluate/pred-data".

<br> 

2. Calculate the scores of various experimental settings on the abilities in the valid or test set:
```bash
time python -u evaluate/scorer.py --pred_data_path evaluate/pred-data/valid #or test
```
&nbsp;&nbsp;&nbsp;&nbsp;The computed scores will be saved in "evaluate/results".

<br> 

3. Plot the graphs as shown in Section 4.3 of the paper.
```bash
python evaluate/plot.py --plot_type # choices=["overall", "curated_vs_synthetic-13b", "ood", "curated_vs_synthetic-7b"]
```
&nbsp;&nbsp;&nbsp;&nbsp;The plotted graphs will be saved in "evaluate/plots".

### Evaluate models on two public benchmarks:

1. Get the dataset from the official repository of [CMMLU](https://github.com/haonan-li/CMMLU/tree/master/data) and [AGIEval](https://github.com/ruixiangcui/AGIEval/tree/main/data) and move their contents to "evaluate/cmmlu/data" and "evaluate/agieval/data" respectively.

2. Calculate the scores in zero-shot and few-shot settings.
```bash
bash evaluate/cmmlu/llama_7b.sh 0 MODEL_NAME_OR_PATH SAVE_TAG
bash evaluate/agieval/llama_7b.sh 0 MODEL_NAME_OR_PATH SAVE_TAG
```
      - 0: Specifying the GPU to be used and the default is 0.
      - MODEL_NAME_OR_PATH: The path for model.
      - SAVE_TAG: The name of the output and log file; for example, "curated-1000_epoch10".
      
## Citation
```
@article{song2023dynamics,
  title={Dynamics of Instruction Tuning: Each Ability of Large Language Models Has Its Own Growth Pace},
  author={Song, Chiyu and Zhou, Zhanchao and Yan, Jianhao and Fei, Yuejiao and Lan, Zhenzhong and Zhang, Yue},
  journal={arXiv preprint arXiv:2310.19651},
  year={2023}
}
```
