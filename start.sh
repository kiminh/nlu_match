# domain识别
start_tm=`date +%s%N`;
export HOST_NAME=$1

# base verson
#export BERT_DIR="/home/${HOST_NAME}/Mywork/model/chinese_L-12_H-768_A-12"
#export CONFIG_FILE=configs/lasertagger_config.json
#export init_checkpoint=${BERT_DIR}/bert_model.ckpt

# large verson
export BERT_DIR="/home/${HOST_NAME}/Mywork/model/roeberta_zh_L-24_H-1024_A-16"
export CONFIG_FILE=configs/lasertagger_config_large.json
export init_checkpoint=${BERT_DIR}/roberta_zh_large_model.ckpt

### Optional parameters ###
# To quickly test that model training works, set the number of epochs to a
# smaller value (e.g. 0.01).
export DOMAIN_NAME="domain"
export input_format="nlu"
export num_train_epochs=4
export TRAIN_BATCH_SIZE=90
export learning_rate=3e-5
export warmup_proportion=0.1
export max_seq_length=45
export drop_keep_prob=0.9
export MAX_INPUT_EXAMPLES=1000000
export SAVE_CHECKPOINT_STEPS=1000
export CORPUS_DIR="/home/${HOST_NAME}/Mywork/corpus/compe/69"
export OUTPUT_DIR="${CORPUS_DIR}/${DOMAIN_NAME}_output"
export MODEL_DIR="${OUTPUT_DIR}/${DOMAIN_NAME}_models"
export do_lower_case=true
export kernel_size=3
export label_map_file=${OUTPUT_DIR}/label_map.json
export SUBMIT_FILE=${MODEL_DIR}/submit.csv
export PREDICTION_FILE=${MODEL_DIR}/pred.tsv

#python preprocess_main.py \
#    --input_file=${CORPUS_DIR}/train.txt \
#    --input_format=${input_format} \
#    --output_tfrecord_train=${OUTPUT_DIR}/train.tf_record \
#    --output_tfrecord_dev=${OUTPUT_DIR}/dev.tf_record \
#    --label_map_file=${label_map_file} \
#    --vocab_file=${BERT_DIR}/vocab.txt \
#    --max_seq_length=${max_seq_length} \
#    --do_lower_case=${do_lower_case}


# Check these numbers from the "*.num_examples" files created in step 2.
export NUM_TRAIN_EXAMPLES=310922
export NUM_EVAL_EXAMPLES=5000

#echo "Train the model."
#python run_lasertagger.py \
#  --training_file=${OUTPUT_DIR}/train.tf_record \
#  --eval_file=${OUTPUT_DIR}/test.tf_record \
#  --label_map_file=${label_map_file} \
#  --model_config_file=${CONFIG_FILE} \
#  --output_dir=${MODEL_DIR} \
#  --init_checkpoint=${BERT_DIR}/bert_model.ckpt \
#  --do_train=true \
#  --do_eval=true \
#  --num_train_epochs=${num_train_epochs} \
#  --train_batch_size=${TRAIN_BATCH_SIZE} \
#  --learning_rate=${learning_rate} \
#  --warmup_proportion=${warmup_proportion} \
#  --drop_keep_prob=${drop_keep_prob} \
#  --kernel_size=${kernel_size}  \
#  --save_checkpoints_steps=${SAVE_CHECKPOINT_STEPS} \
#  --max_seq_length=${max_seq_length} \
#  --num_train_examples=${NUM_TRAIN_EXAMPLES} \
#  --num_eval_examples=${NUM_EVAL_EXAMPLES}



### 4. Prediction

# Export the model.
echo "Export the model."
python run_lasertagger.py \
  --label_map_file=${label_map_file} \
  --model_config_file=${CONFIG_FILE} \
  --max_seq_length=${max_seq_length} \
  --kernel_size=${kernel_size}  \
  --output_dir=${MODEL_DIR}/ \
  --do_export=true \
  --export_path=${MODEL_DIR}/export


######## Get the most recently exported model directory.
TIMESTAMP=$(ls "${MODEL_DIR}/export/" | \
            grep -v "temp-" | sort -r | head -1)
SAVED_MODEL_DIR=${MODEL_DIR}/export/${TIMESTAMP}


echo "predict_main.py for eval"
python predict_main.py \
  --input_file=${CORPUS_DIR}/dev.txt \
  --input_format=${input_format} \
  --output_file=${PREDICTION_FILE} \
  --label_map_file=${label_map_file} \
  --vocab_file=${BERT_DIR}/vocab.txt \
  --max_seq_length=${max_seq_length} \
  --do_lower_case=${do_lower_case} \
  --saved_model=${SAVED_MODEL_DIR} \
  --domain_score_folder=${MODEL_DIR}


##### 5. Evaluation
echo "python score_main.py --prediction_file=" ${PREDICTION_FILE}
python score_main.py --prediction_file=${PREDICTION_FILE} --do_lower_case=true


#echo "predict_main.py for test"
#python predict_main.py \
#  --input_file=${CORPUS_DIR}/test.csv \
#  --input_format=${input_format} \
#  --output_file=${PREDICTION_FILE} \
#  --submit_file=${SUBMIT_FILE} \
#  --label_map_file=${label_map_file} \
#  --vocab_file=${BERT_DIR}/vocab.txt \
#  --max_seq_length=${max_seq_length} \
#  --do_lower_case=${do_lower_case} \
#  --saved_model=${SAVED_MODEL_DIR} \
#  --domain_score_folder=${MODEL_DIR}


end_tm=`date +%s%N`;
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000 /3600}'`
echo "cost time" $use_tm "h"
