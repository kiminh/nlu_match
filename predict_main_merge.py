# coding=utf-8
# 模型融合

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
flags.DEFINE_bool(
    'do_lower_case', False,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
flags.DEFINE_string('saved_model', None, 'Path to an exported TF model.')

flags.DEFINE_string('domain_score_folder', None, 'Path to save domain_score.')
flags.DEFINE_string('prevous_domain_scores', None, 'Path to save domain_score.')

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('output_file')
    flags.mark_flag_as_required('label_map_file')


    ##### test
    print(colored("%s input file:%s" % (curLine(), FLAGS.input_file), "red"))
    domain_list = []
    slot_info_list = []
    intent_list = []
    sources_list = []
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

    number = len(sources_list)  # 总样本数
    all_pred_domainMap_list = []
    for prevous_domain_scores in FLAGS.prevous_domain_scores.split(";"):
        if FLAGS.submit_file is not None:
            domain_score_file = "%s/submit_domain_score%s.json" % (FLAGS.domain_score_folder, prevous_domain_scores)
        else:
            domain_score_file = "%s/predict_domain_score%s.json" % (FLAGS.domain_score_folder, prevous_domain_scores)
        with open(domain_score_file, "r") as fr:
            pred_domainMap_list = json.load(fr)
            assert len(pred_domainMap_list) == number
        print(curLine(), "load %d from %s" % (len(pred_domainMap_list), domain_score_file))
        all_pred_domainMap_list.append((float(prevous_domain_scores), pred_domainMap_list))



    predict_domain_list = []
    predict_intent_list = []
    predict_slot_list = []
    start_time = time.time()
    num_predicted = 0
    modemode = 'a'
    if len(domain_list) > 0:  # 有标注
        modemode = 'w'


    print(curLine(), FLAGS.output_file)
    with tf.gfile.Open(FLAGS.output_file, modemode) as writer:
        if len(domain_list) > 0:  # 有标注
            writer.write("\t".join(["sessionId", "query", "predDomain", "predIntent", "predSlot", "domain", "intent", "Slot"]) + "\n")
        previous_sessionId = None
        domain_history = []
        for exampleId in range(number):
            # merge score
            merge_pred_domainMap = {}
            for prevous_domain_scores,pred_domainMap_list in all_pred_domainMap_list:
                pred_domainMap = pred_domainMap_list[exampleId]
                for domain, score in pred_domainMap.items():
                    if domain in merge_pred_domainMap:
                        merge_pred_domainMap[domain] += score * prevous_domain_scores
                    else:
                        merge_pred_domainMap[domain] = score * prevous_domain_scores
            current_predict_domain = None
            current_predict_score = 0
            for domain, score in merge_pred_domainMap.items():
                if score > current_predict_score:
                    current_predict_score = score
                    current_predict_domain = domain
            current_predict_score = current_predict_score/len(all_pred_domainMap_list)
            num_predicted += 1
            sessionId, raw_query = session_list[exampleId]
            if sessionId != previous_sessionId:  # 新的会话
                domain_history = []
                previous_sessionId = sessionId
            predict_domain, predict_intent, slot_info = rules(raw_query, current_predict_domain, domain_history)
            domain_history.append((predict_domain,predict_intent)) # 记录多轮
            predict_domain_list.append(predict_domain)
            predict_intent_list.append(predict_intent)
            predict_slot_list.append(slot_info)
            if len(domain_list)>0:  # 有标注
                domain = domain_list[exampleId]
                intent = intent_list[exampleId]
                slot = slot_info_list[exampleId]
                writer.write("\t".join([sessionId, raw_query, predict_domain, predict_intent, slot_info, domain, intent, slot]) + "\n")
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
