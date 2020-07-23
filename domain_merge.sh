# 模型融合ｄｏｍａｉｎ分类

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
# large verson
export BERT_DIR="/home/${HOST_NAME}/Mywork/model/roeberta_zh_L-24_H-1024_A-16"
export CONFIG_FILE=configs/lasertagger_config_large.json
export init_checkpoint=${BERT_DIR}/roberta_zh_large_model.ckpt

### Optional parameters ###
# To quickly test that model training works, set the number of epochs to a
# smaller value (e.g. 0.01).
export DOMAIN_NAME="domain"
export MAX_INPUT_EXAMPLES=1000000
export SAVE_CHECKPOINT_STEPS=1000
export CORPUS_DIR="/home/${HOST_NAME}/Mywork/corpus/compe/69"
export OUTPUT_DIR="${CORPUS_DIR}/${DOMAIN_NAME}_output"
export MODEL_DIR="${OUTPUT_DIR}/${DOMAIN_NAME}_models"
export do_lower_case=true
export label_map_file=${OUTPUT_DIR}/label_map.json
export SUBMIT_FILE=${MODEL_DIR}/submit.csv
export PREDICTION_FILE=${MODEL_DIR}/pred.tsv
export prevous_domain_scores="93.03;92.91;93.21"


echo "predict_main.py for eval"
python predict_main_merge.py \
  --input_file=${CORPUS_DIR}/dev.txt \
  --output_file=${PREDICTION_FILE} \
  --label_map_file=${label_map_file} \
  --do_lower_case=${do_lower_case} \
  --domain_score_folder=${OUTPUT_DIR} \
  --prevous_domain_scores=${prevous_domain_scores}

##### 5. Evaluation
echo "python score_main.py --prediction_file=" ${PREDICTION_FILE}
python score_main.py --prediction_file=${PREDICTION_FILE} --do_lower_case=true


echo "predict_main.py for test"
python predict_main_merge.py \
  --input_file=${CORPUS_DIR}/test.csv \
  --output_file=${PREDICTION_FILE} \
  --submit_file=${SUBMIT_FILE} \
  --label_map_file=${label_map_file} \
  --do_lower_case=${do_lower_case} \
  --domain_score_folder=${OUTPUT_DIR} \
  --prevous_domain_scores=${prevous_domain_scores}


end_tm=`date +%s%N`;
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000 /3600}'`
echo "cost time" $use_tm "h"
