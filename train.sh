# 文本摘要任务
# python -m pip install --upgrade pip -i https://pypi.douban.com/simple
# pip install -i https://pypi.douban.com/simple/ bert-tensorflow==1.0.1
# pip install -i https://pypi.douban.com/simple/ tensorflow==1.15.0

start_tm=`date +%s%N`;

export HOST_NAME=$1
if [[ "wzk" == "$HOST_NAME" ]]
then
  # set gpu id to use
  export CUDA_VISIBLE_DEVICES=0
else
  # not use gpu
  export CUDA_VISIBLE_DEVICES=""
fi

# base verson
#export BERT_DIR="/home/${HOST_NAME}/Mywork/model/chinese_L-12_H-768_A-12"
#export CONFIG_FILE=configs/lasertagger_config.json
#export init_checkpoint=${BERT_DIR}/bert_model.ckpt
#export TRAIN_BATCH_SIZE=50
#export learning_rate=1.5e-5

# large verson
export BERT_DIR="/home/${HOST_NAME}/Mywork/model/roeberta_zh_L-24_H-1024_A-16"
export CONFIG_FILE=configs/lasertagger_config_large.json
export init_checkpoint=${BERT_DIR}/roberta_zh_large_model.ckpt
export TRAIN_BATCH_SIZE=30
export learning_rate=1.2e-5

### Optional parameters ###
# To quickly test that model training works, set the number of epochs to a
# smaller value (e.g. 0.01).
export DOMAIN_NAME="domain"
export input_format="nlu"
export num_train_epochs=5



#export num_train_epochs=5
#export TRAIN_BATCH_SIZE=100
#export learning_rate=3e-5

export warmup_proportion=0.1
export max_seq_length=45
export drop_keep_prob=0.8
export MAX_INPUT_EXAMPLES=1000000
export SAVE_CHECKPOINT_STEPS=2000
export CORPUS_DIR="/home/${HOST_NAME}/Mywork/corpus/compe/69"

export OUTPUT_DIR="${CORPUS_DIR}/${DOMAIN_NAME}_output"
export do_lower_case=true
export kernel_size=3
export label_map_file=${OUTPUT_DIR}/label_map.json
export SUBMIT_FILE=${OUTPUT_DIR}/models/submit.csv

# Check these numbers from the "*.num_examples" files created in step 2.
export NUM_TRAIN_EXAMPLES=300000
export NUM_EVAL_EXAMPLES=5000

#python preprocess_main.py \
#    --input_file=${CORPUS_DIR}/train.txt \
#    --input_format=${input_format} \
#    --output_tfrecord_train=${OUTPUT_DIR}/train.tf_record \
#    --output_tfrecord_dev=${OUTPUT_DIR}/dev.tf_record \
#    --label_map_file=${label_map_file} \
#    --vocab_file=${BERT_DIR}/vocab.txt \
#    --max_seq_length=${max_seq_length} \
#    --do_lower_case=${do_lower_case}


echo "Train the model."
python run_lasertagger.py \
  --training_file=${OUTPUT_DIR}/train.tf_record \
  --eval_file=${OUTPUT_DIR}/test.tf_record \
  --label_map_file=${label_map_file} \
  --model_config_file=${CONFIG_FILE} \
  --output_dir=${OUTPUT_DIR}/${DOMAIN_NAME}_models \
  --init_checkpoint=${init_checkpoint} \
  --do_train=true \
  --do_eval=true \
  --num_train_epochs=${num_train_epochs} \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --learning_rate=${learning_rate} \
  --warmup_proportion=${warmup_proportion} \
  --drop_keep_prob=${drop_keep_prob} \
  --kernel_size=${kernel_size}  \
  --save_checkpoints_steps=${SAVE_CHECKPOINT_STEPS} \
  --max_seq_length=${max_seq_length} \
  --num_train_examples=${NUM_TRAIN_EXAMPLES} \
  --num_eval_examples=${NUM_EVAL_EXAMPLES}



### 4. Prediction

### Export the model.
#echo "Export the model."
#python run_lasertagger.py \
#  --label_map_file=${label_map_file} \
#  --model_config_file=${CONFIG_FILE} \
#  --max_seq_length=${max_seq_length} \
#  --kernel_size=${kernel_size}  \
#  --output_dir=${OUTPUT_DIR}/${DOMAIN_NAME}_models/ \
#  --do_export=true \
#  --export_path=${OUTPUT_DIR}/models/export
#
#
######### Get the most recently exported model directory.
#TIMESTAMP=$(ls "${OUTPUT_DIR}/models/export/" | \
#            grep -v "temp-" | sort -r | head -1)
#SAVED_MODEL_DIR=${OUTPUT_DIR}/models/export/${TIMESTAMP}
#PREDICTION_FILE=${OUTPUT_DIR}/models/pred.tsv
#
#echo "predict_main.py for eval"
#python predict_main.py \
#  --input_file=${CORPUS_DIR}/dev.txt \
#  --input_format=${input_format} \
#  --output_file=${PREDICTION_FILE} \
#  --label_map_file=${label_map_file} \
#  --vocab_file=${BERT_DIR}/vocab.txt \
#  --max_seq_length=${max_seq_length} \
#  --do_lower_case=${do_lower_case} \
#  --saved_model=${SAVED_MODEL_DIR}
#
###### 5. Evaluation
#echo "python score_main.py --prediction_file=" ${PREDICTION_FILE}
#python score_main.py --prediction_file=${PREDICTION_FILE} --vocab_file=${BERT_DIR}/vocab.txt --do_lower_case=true


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
#  --saved_model=${SAVED_MODEL_DIR}


end_tm=`date +%s%N`;
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000 /3600}'`
echo "cost time" $use_tm "h"