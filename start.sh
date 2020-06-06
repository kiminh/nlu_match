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

### Optional parameters ###
# To quickly test that model training works, set the number of epochs to a
# smaller value (e.g. 0.01).
export input_format="qa"
export num_train_epochs=3
export TRAIN_BATCH_SIZE=48
export learning_rate=3e-5
export warmup_proportion=0.2
export PHRASE_VOCAB_SIZE=2000
export max_seq_length=300
export drop_keep_prob=0.9
export MAX_INPUT_EXAMPLES=1000000
export SAVE_CHECKPOINT_STEPS=1000
export enable_swap_tag=false
export output_arbitrary_targets_for_infeasible_examples=false
export CORPUS_DIR="/home/${HOST_NAME}/Mywork/corpus/summary/ms_thu" # _google
export BERT_BASE_DIR="/home/${HOST_NAME}/Mywork/model/RoBERTa-tiny-clue"
export OUTPUT_DIR="${CORPUS_DIR}/output"
export do_lower_case=true
export kernel_size=3

#python phrase_vocabulary_optimization.py \
#  --input_file=${CORPUS_DIR}/train.txt \
#  --input_format=${input_format} \
#  --vocabulary_size=${PHRASE_VOCAB_SIZE} \
#  --max_input_examples=3000000 \
#  --enable_swap_tag=${enable_swap_tag} \
#  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
#  --do_lower_case=${do_lower_case} \
#  --output_file=${OUTPUT_DIR}/label_map.txt

#python preprocess_main.py \
#    --input_file=${CORPUS_DIR}/train.txt \
#    --input_format=${input_format} \
#    --output_tfrecord=${OUTPUT_DIR}/train.tf_record \
#    --label_map_file=${OUTPUT_DIR}/label_map.txt \
#    --vocab_file=${BERT_BASE_DIR}/vocab.txt \
#    --max_seq_length=${max_seq_length} \
#    --enable_swap_tag=${enable_swap_tag} \
#    --do_lower_case=${do_lower_case} \
#    --output_arbitrary_targets_for_infeasible_examples=${output_arbitrary_targets_for_infeasible_examples}

#echo "python preprocess_main.py"
#python preprocess_main.py \
#  --input_file=${CORPUS_DIR}/dev.txt \
#  --input_format=${input_format} \
#  --output_tfrecord=${OUTPUT_DIR}/tune.tf_record \
#  --label_map_file=${OUTPUT_DIR}/label_map.txt \
#  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
#  --max_seq_length=${max_seq_length} \
#  --enable_swap_tag=${enable_swap_tag} \
#  --do_lower_case=${do_lower_case} \
#  --output_arbitrary_targets_for_infeasible_examples=${output_arbitrary_targets_for_infeasible_examples}

#echo "python preprocess_main.py"
#python preprocess_main.py \
#  --input_file=/home/${HOST_NAME}/Mywork/corpus/summary/yunying/yunying_annotate.txt \
#  --input_format=${input_format} \
#  --output_tfrecord=${OUTPUT_DIR}/test.tf_record \
#  --label_map_file=${OUTPUT_DIR}/label_map.txt \
#  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
#  --max_seq_length=${max_seq_length} \
#  --enable_swap_tag=${enable_swap_tag} \
#  --do_lower_case=${do_lower_case} \
#  --output_arbitrary_targets_for_infeasible_examples=${output_arbitrary_targets_for_infeasible_examples}






# Check these numbers from the "*.num_examples" files created in step 2.
export NUM_TRAIN_EXAMPLES=310922
export NUM_EVAL_EXAMPLES=5000
export CONFIG_FILE=configs/lasertagger_config.json



#echo "Train the model."
#python run_lasertagger.py \
#  --training_file=${OUTPUT_DIR}/train.tf_record \
#  --eval_file=${OUTPUT_DIR}/test.tf_record \
#  --label_map_file=${OUTPUT_DIR}/label_map.txt \
#  --model_config_file=${CONFIG_FILE} \
#  --output_dir=${OUTPUT_DIR}/models \
#  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
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

### Export the model.
echo "Export the model."
python run_lasertagger.py \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --model_config_file=${CONFIG_FILE} \
  --max_seq_length=${max_seq_length} \
  --kernel_size=${kernel_size}  \
  --output_dir=${OUTPUT_DIR}/models/ \
  --do_export=true \
  --export_path=${OUTPUT_DIR}/models/export


######## Get the most recently exported model directory.
TIMESTAMP=$(ls "${OUTPUT_DIR}/models/export/" | \
            grep -v "temp-" | sort -r | head -1)
SAVED_MODEL_DIR=${OUTPUT_DIR}/models/export/${TIMESTAMP}
PREDICTION_FILE=${OUTPUT_DIR}/models/pred.tsv

echo "predict_main.py"
export test_file=/home/${HOST_NAME}/Mywork/corpus/summary/yunying/yunying_annotate.txt
#  test_file=${CORPUS_DIR}/dev.txt
python predict_main.py \
  --input_file=${test_file} \
  --input_format=wikisplit \
  --output_file=${PREDICTION_FILE} \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --max_seq_length=${max_seq_length} \
  --do_lower_case=${do_lower_case} \
  --saved_model=${SAVED_MODEL_DIR}
#
#
###### 5. Evaluation
echo "python score_main.py --prediction_file=" ${PREDICTION_FILE}
python score_main.py --prediction_file=${PREDICTION_FILE} --vocab_file=${BERT_BASE_DIR}/vocab.txt --do_lower_case=true

end_tm=`date +%s%N`;
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000 /3600}'`
echo "cost time" $use_tm "h"
