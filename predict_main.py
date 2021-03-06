# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Compute realized predictions for a dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import collections, os
import math, time
import json
from termcolor import colored
import tensorflow as tf

import bert_example
import predict_utils
import csv
import utils
import acmation
from curLine_file import curLine, normal_transformer, other_tag

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'input_file', None,
    'Path to the input file containing examples for which to compute '
    'predictions.')
flags.DEFINE_enum(
    'input_format', None, ['wikisplit', 'discofuse','qa', 'nlu'],
    'Format which indicates how to parse the input_file.')
flags.DEFINE_string(
    'output_file', None,
    'Path to the CSV file where the predictions are written to.')
flags.DEFINE_string(
    'submit_file', None,
    'Path to the CSV file where the predictions are written to.')
flags.DEFINE_string(
    'label_map_file', None,
    'Path to the label map file. Either a JSON file ending with ".json", that '
    'maps each possible tag to an ID, or a text file that has one tag per '
    'line.')
flags.DEFINE_string('vocab_file', None, 'Path to the BERT vocabulary file.')
flags.DEFINE_integer('max_seq_length', 128, 'Maximum sequence length.')
flags.DEFINE_bool(
    'do_lower_case', False,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
flags.DEFINE_string('saved_model', None, 'Path to an exported TF model.')

flags.DEFINE_string('domain_score_folder', None, 'Path to save domain_score.')

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('input_format')
    flags.mark_flag_as_required('output_file')
    flags.mark_flag_as_required('label_map_file')
    flags.mark_flag_as_required('vocab_file')
    flags.mark_flag_as_required('saved_model')
    label_map = utils.read_label_map(FLAGS.label_map_file)
    builder = bert_example.BertExampleBuilder(label_map, FLAGS.vocab_file,
                                              FLAGS.max_seq_length,
                                              FLAGS.do_lower_case)
    predictor = predict_utils.LaserTaggerPredictor(
        tf.contrib.predictor.from_saved_model(FLAGS.saved_model), builder,
        label_map)
    print(colored("%s saved_model:%s" % (curLine(), FLAGS.saved_model), "red"))

    ##### test
    print(colored("%s input file:%s" % (curLine(), FLAGS.input_file), "red"))
    domain_list = []
    slot_info_list = []
    intent_list = []
    sources_list = []
    predict_batch_size = 32
    limit = predict_batch_size * 1500 # 5184 #　10001 #
    with tf.gfile.GFile(FLAGS.input_file) as f:
        reader = csv.reader(f)
        session_list = []
        for row_id, line in enumerate(reader):
            if len(line) > 2:
                (sessionId, raw_query, domain_intent, slot) = line
            else:
                (sessionId, raw_query) = line
            query = normal_transformer(raw_query)
            sources = []
            if row_id > 1 and sessionId == session_list[row_id - 2][0]:
                sources.append(session_list[row_id - 2][1])  # last last query
            if row_id > 0 and sessionId == session_list[row_id - 1][0]:
                sources.append(session_list[row_id - 1][1])  # last query
            sources.append(query)
            session_list.append((sessionId, raw_query))
            sources_list.append(sources)
            if len(line) > 2:  # 有标注
                if domain_intent == other_tag:
                    domain = other_tag
                    intent = other_tag
                else:
                    domain, intent = domain_intent.split(".")
                domain_list.append(domain)
                intent_list.append(intent)
                slot_info_list.append(slot)
            if len(sources_list) >= limit:
                print(colored("%s stop reading at %d to save time" %(curLine(), limit), "red"))
                break

    number = len(sources_list)  # 总样本数
    predict_domain_list = []
    predict_intent_list = []
    predict_slot_list = []
    pred_domainMap_list = []
    predict_batch_size = min(predict_batch_size, number)
    batch_num = math.ceil(float(number) / predict_batch_size)
    start_time = time.time()
    num_predicted = 0
    modemode = 'a'
    if len(domain_list) > 0:  # 有标注
        modemode = 'w'
    previous_sessionId = None
    domain_history = []
    with tf.gfile.Open(FLAGS.output_file, modemode) as writer:
        if len(domain_list) > 0:  # 有标注
            writer.write("\t".join(["sessionId", "query", "predDomain", "predIntent", "predSlot", "domain", "intent", "Slot"]) + "\n")
        for batch_id in range(batch_num):
            sources_batch = sources_list[batch_id * predict_batch_size: (batch_id + 1) * predict_batch_size]
            prediction_batch, pred_domainMap_batch = predictor.predict_batch(sources_batch=sources_batch)
            assert len(prediction_batch) == len(sources_batch)
            num_predicted += len(prediction_batch)
            for id, [current_predict_domain, pred_domainMap, sources] in enumerate(zip(prediction_batch, pred_domainMap_batch, sources_batch)):
                sessionId, raw_query = session_list[batch_id * predict_batch_size + id]
                if sessionId != previous_sessionId:  # 新的会话
                    domain_history = []
                    previous_sessionId = sessionId
                predict_domain, predict_intent, slot_info = rules(raw_query, current_predict_domain, domain_history)
                pred_domainMap_list.append(pred_domainMap)
                domain_history.append((predict_domain,predict_intent)) # 记录多轮
                predict_domain_list.append(predict_domain)
                predict_intent_list.append(predict_intent)
                predict_slot_list.append(slot_info)
                if len(domain_list)>0:  # 有标注
                    domain = domain_list[batch_id * predict_batch_size + id]
                    intent = intent_list[batch_id * predict_batch_size + id]
                    slot = slot_info_list[batch_id * predict_batch_size + id]
                    writer.write("\t".join([sessionId, raw_query, predict_domain, predict_intent, slot_info, domain, intent, slot]) + "\n")
            if batch_id % 5 == 0:
                cost_time = (time.time() - start_time) / 60.0
                print("%s batch_id=%d/%d, predict %d/%d examples, cost %.2fmin." %
                      (curLine(), batch_id + 1, batch_num, num_predicted, number, cost_time))
    cost_time = (time.time() - start_time) / 60.0
    print(
        f'{curLine()} {num_predicted} predictions saved to:{FLAGS.output_file}, cost {cost_time} min, ave {cost_time/num_predicted*60} s.')


    if FLAGS.submit_file is not None:
        domain_counter = collections.Counter()
        if os.path.exists(path=FLAGS.submit_file):
            os.remove(FLAGS.submit_file)
        with open(FLAGS.submit_file, 'w',encoding='UTF-8') as f:
            writer = csv.writer(f, dialect='excel')
            # writer.writerow(["session_id", "query", "intent", "slot_annotation"])  # TODO
            for example_id, sources in enumerate(sources_list):
                sessionId, raw_query = session_list[example_id]
                predict_domain = predict_domain_list[example_id]
                predict_intent = predict_intent_list[example_id]
                predict_domain_intent = other_tag
                domain_counter.update([predict_domain])
                if predict_domain != other_tag:
                    predict_domain_intent = "%s.%s" % (predict_domain, predict_intent)
                line = [sessionId, raw_query, predict_domain_intent, predict_slot_list[example_id]]
                writer.writerow(line)
        print(curLine(), "example_id=", example_id)
        print(curLine(), "domain_counter:", domain_counter)
        cost_time = (time.time() - start_time) / 60.0
        num_predicted = example_id+1
        print(curLine(), "domain cost %f s" % (cost_time))
        print(
            f'{curLine()} {num_predicted} predictions saved to:{FLAGS.submit_file}, cost {cost_time} min, ave {cost_time/num_predicted*60} s.')
        domain_score_file = "%s/submit_domain_score.json" % (FLAGS.domain_score_folder)
    else:
        domain_score_file = "%s/predict_domain_score.json" % (FLAGS.domain_score_folder)

    with open(domain_score_file, "w") as fw:
        json.dump(pred_domainMap_list, fw, ensure_ascii=False, indent=4)
    print(curLine(), "dump %d to %s" % (len(pred_domainMap_list), domain_score_file))

def rules(raw_query, predict_domain, domain_history):
    slot_info = raw_query
    cancel_intent = False
    for word in predict_utils.cancel_keywords:
        if word in raw_query:
            cancel_intent = True
            break
    if cancel_intent:
        if "导航" in raw_query or "路况" in raw_query:
            predict_domain = 'navigation'
        elif "音乐" in raw_query or "音响" in raw_query or "播放" in raw_query or "music" in raw_query or "媒体" in raw_query or "mp3" in raw_query or "歌" in raw_query or "曲" in raw_query:
            predict_domain = 'music'
        elif "电话" in raw_query or "拨打" in raw_query or "联系" in raw_query:
            predict_domain = 'phone_call'
        elif "3g" in raw_query or "网络" in raw_query or "电子狗" in raw_query or "屏幕" in raw_query or "收音机" in raw_query:
            predict_domain = other_tag
        else:
            current_domain = other_tag
            for dom,int in domain_history[::-1]:  # 逆序
                if dom in {'navigation', 'music', 'phone_call'}: # TODO
                    if dom not in {'navigation', 'music'} and ("cancel" in int or "pause" in int):
                        current_domain = other_tag  #  验证集上有些样本不符合这个规则
                        print(curLine(), current_domain, predict_domain, raw_query)
                    else:
                        current_domain = dom
                    break
            if predict_domain != current_domain:
                predict_domain = current_domain

    predict_intent = predict_domain
    if predict_domain == "navigation":
        predict_intent = 'navigation'
        if "打开" in raw_query:
            predict_intent = "open"
        elif "开始" in raw_query:
            predict_intent = "start_navigation"
        for word in predict_utils.cancel_keywords:
            if word in raw_query:
                predict_intent = 'cancel_navigation'
                break
        slot_info = acmation.get_slot_info(raw_query, domain=predict_domain)
    elif predict_domain == 'music':
        predict_intent = 'play'
        for word in predict_utils.cancel_keywords:
            if word in raw_query:
                predict_intent = 'pause'
                break
        for word in ["下一", "换一首", "换一曲", "切歌", "其他歌"]:
            if word in raw_query:
                predict_intent = 'next'
                break
        slot_info = acmation.get_slot_info(raw_query, domain=predict_domain)
        if predict_intent not in ['play','pause'] and slot_info != raw_query: # 根据槽位修改意图　　换一首<singer>高安</singer>的<song>红尘情歌</song>
            print(curLine(), predict_intent, slot_info)
            predict_intent = 'play'
        # if predict_intent != 'play': # 换一首<singer>高安</singer>的<song>红尘情歌</song>
        #     print(curLine(), predict_intent, slot_info)
    elif predict_domain == 'phone_call':
        predict_intent = 'make_a_phone_call'
        for word in predict_utils.cancel_keywords:
            if word in raw_query:
                predict_intent = 'cancel'
                break
        slot_info = acmation.get_slot_info(raw_query, domain=predict_domain)
    return predict_domain, predict_intent, slot_info

if __name__ == '__main__':
    app.run(main)
