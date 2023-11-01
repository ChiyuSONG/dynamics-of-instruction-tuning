# Dynamics of Instruction Tuning: Each Ability of Large Language Models Has Its Own Growth Pace (WIP)

This [paper](https://arxiv.org/abs/2310.19651) investigated how the underlying abilities of Large Language Models (LLMs), such as creative writing, code generation, and logical reasoning, develop at varying paces during instruction tuning. We systematically studied the effects of data volume, parameter size (7b-33b), and data construction methods on the growth of each ability.

- The codebase and commands are provided to reproduce our experimental results.
- The human-curated dataset for training and evaluation can be found [here](https://huggingface.co/datasets/ChiyuSONG/dynamics-of-instruction-tuning).

<p align="center" width="100%">
      <img src="img/way_to_agi.jpg" alt="Each ability of LLMs has its own growth pace during instruction tuning." style="width: 50%; min-width: 200px; display: block; margin: auto;">
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
bash run_train.sh --data_type curated-160 --model_size 7b --model_name_or_path **path_to_foundation_model** --batch_size 8 --gradient_accumulation 1
```
    Training logs and model checkpoints will be saved in "\runs".

## Evaluation
### Evaluate models on human-curated valid/test sets:

1. Generate predictions on the valid/test questions.
```bash
export CUDA_VISIBLE_DEVICES=0
time python -u evaluate/pred.py --model_name_or_path **path_to_saved_checkpoint** --eval_data_path data/curated/valid #or test
```
    The generated answers will be saved in "evaluate/pred-data".

2. Calculate the scores of various experimental settings on the abilities in the valid or test set:
```bash
time python -u evaluate/scorer.py --pred_data_path evaluate/pred-data/valid #or test
```
    The computed scores will be saved in "evaluate/results".

3. Plot the graphs as shown in Section 4.3 of the paper.
```bash
python evaluate/plot.py --plot_type # choices=["overall", "curated_vs_synthetic-13b", "ood", "curated_vs_synthetic-7b"]
```
    The plotted graphs will be saved in "evaluate/plots".


## Citation
```
@ARTICLE{2023arXiv231019651S,
       author = {{Song}, Chiyu and {Zhou}, Zhanchao and {Yan}, Jianhao and {Fei}, Yuejiao and {Lan}, Zhenzhong and {Zhang}, Yue},
        title = "{Dynamics of Instruction Tuning: Each Ability of Large Language Models Has Its Own Growth Pace}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computation and Language},
         year = 2023,
        month = oct,
          eid = {arXiv:2310.19651},
        pages = {arXiv:2310.19651},
archivePrefix = {arXiv},
       eprint = {2310.19651},
 primaryClass = {cs.CL},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv231019651S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
