# Dynamics of Instruction Fine-Tuning for Chinese Large Language Models

This [paper](https://arxiv.org/abs/2310.19651) explored how data quantity, model size, and data construction methods affect the development of different abilities in Chinese large language models using a new dataset called *DoIT*. We found that abilities respond differently to scaling factors and identified two key features—Complexity and Transference—that predict ability growth. This insight helps optimize training strategies and improve model performance on different benchmarks.

- The codebase and commands are provided to reproduce our experimental results.
- The human-curated dataset, *DoIT*, for training and evaluation can be found [here](https://huggingface.co/datasets/ChiyuSONG/dynamics-of-instruction-tuning).

<p align="center" width="100%">
      <img src="img/way_to_agi.jpg" alt="Each ability of LLMs has its own growth pace during instruction tuning." style="width: 56%; min-width: 200px; display: block; margin: auto;">
</p>

## Dependencies
The code is implemented using python 3.9 and PyTorch v2.0.1
```bash
pip install -r requirements.txt
```

## Training
1. Get the *DoIT* dataset and move its content to "data/":
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
@inproceedings{song2025dynamics,
  title={Dynamics of Instruction Fine-Tuning for Chinese Large Language Models},
  author={Song, Chiyu and Zhou, Zhanchao and Yan, Jianhao and Fei, Yuejiao and Lan, Zhenzhong and Zhang, Yue},
  booktitle={Proceedings of the 31st International Conference on Computational Linguistics},
  pages={10345--10366},
  year={2025}
}
```
