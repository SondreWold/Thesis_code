ROOT_PATH="./data/concept_net/predicate_pre/corpora/"

TOKENIZER="bert-base-uncased"
MODEL_TYPE="bert-base-uncased"
TRAINING_FOLDER="./src"
OUTPUT_DIR="./models/houlsby_original_uncased"
ADAPTER_OUTPUT="./full_run/adapters/"


for STEP in "atLocation" "capableOf" "causes" "causesDesire""desires" "hasA" "hasPrerequisite" "hasProperty" "hasSubevent" "isA" "locatedNear" "madeOf" "motivatedByGoal" "partOf" "receivesAction" "usedFor"; do
    echo $ROOT_PATH + $STEP + "/" + $STEP + "_train.txt"
    python $TRAINING_FOLDER/run_mlm.py \
        --model_name_or_path $ROOT_PATH + $STEP + "/" + $STEP + "_train.txt" \
        --train_file $ROOT_PATH + $STEP + "/" + $STEP + "_val.txt" \
        --validation_file $VAL_FILE \
        --output_dir $OUTPUT_DIR \
        --line_by_line True \
        --adapter_name $STEP \
        --only_save_adapter \
        --single_adapter_path $ADAPTER_OUTPUT + $STEP
        --adapter_config "houlsby" \
        --non_linearity "gelu" \
        --reduction_factor 12   \
        --num_warmup_steps 10000 \
        --max_train_steps 100000 
