cd evaluate/agieval

export CUDA_VISIBLE_DEVICES=$1
MODEL_NAME_OR_PATH=$2
SAVE_TAG=$3

mkdir -p logs
mkdir -p outputs

for i in z f; do
python run_prediction_llama.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir outputs/$i.$SAVE_TAG \
    --dataset_dir data/v1 \
    --raw_prompt_path data/few_shot_prompts.csv \
    --setting_name $i \
    2>&1 | tee logs/$i.$SAVE_TAG.log
done
