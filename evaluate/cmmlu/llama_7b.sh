cd evaluate/cmmlu

export CUDA_VISIBLE_DEVICES=$1
MODEL_NAME_OR_PATH=$2
SAVE_TAG=$3

mkdir -p logs
mkdir -p outputs

for i in 0 5; do
python run_prediction_llama.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_dir data \
    --save_dir outputs/$SAVE_TAG \
    --num_few_shot $i \
    --sample_ratio 1 \
    2>&1 | tee logs/$i.$SAVE_TAG
done