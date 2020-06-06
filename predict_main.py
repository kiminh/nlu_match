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
from absl import logging
import math, time
from termcolor import colored
import tensorflow as tf

import bert_example
import predict_utils
import tagging_converter
import utils
from bert import tokenization
from curLine_file import curLine

sep_str = '$'
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'input_file', None,
    'Path to the input file containing examples for which to compute '
    'predictions.')
flags.DEFINE_enum(
    'input_format', None, ['wikisplit', 'discofuse'],
    'Format which indicates how to parse the input_file.')
flags.DEFINE_string(
    'output_file', None,
    'Path to the TSV file where the predictions are written to.')
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
flags.DEFINE_bool('enable_swap_tag', True, 'Whether to enable the SWAP tag.')
flags.DEFINE_string('saved_model', None, 'Path to an exported TF model.')


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
    tokenizer = tokenization.FullTokenizer(FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    converter = tagging_converter.TaggingConverter(
        tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
        FLAGS.enable_swap_tag, tokenizer=tokenizer)
    builder = bert_example.BertExampleBuilder(label_map, FLAGS.vocab_file,
                                              FLAGS.max_seq_length,
                                              FLAGS.do_lower_case, converter)
    predictor = predict_utils.LaserTaggerPredictor(
        tf.contrib.predictor.from_saved_model(FLAGS.saved_model), builder,
        label_map)
    print(colored("%s saved_model:%s" % (curLine(), FLAGS.saved_model), "red"))

    ##### test
    print(colored("%s input file:%s" % (curLine(), FLAGS.input_file), "red"))
    sources_list = []
    target_list = []
    predict_batch_size = 64
    limit = predict_batch_size * 15 * 2 # 5184 #　10001 #
    with tf.gfile.GFile(FLAGS.input_file) as f:
        for line in f:
            line_split = line.rstrip('\n').split('\t')
            if len(line_split) != 4:
                print(curLine(), "ignore %d line_split:" % (len(line_split)), line_split)
                continue
            question, sources, target, lcs_rate = line_split
            sources_list.append([question, sources])
            target_list.append(target)
            if len(sources_list) >= limit:
                print(colored("%s stop reading at %d to save time" %(curLine(), limit), "red"))
                break
    number = len(sources_list)  # 总样本数
    predict_batch_size = min(predict_batch_size, number)
    batch_num = math.ceil(float(number) / predict_batch_size)
    start_time = time.time()
    num_predicted = 0
    with tf.gfile.Open(FLAGS.output_file, 'w') as writer:
        writer.write(f'source\tprediction\ttarget\n')
        for batch_id in range(batch_num):
            sources_batch = sources_list[batch_id * predict_batch_size: (batch_id + 1) * predict_batch_size]
            prediction_batch = predictor.predict_batch(sources_batch=sources_batch)
            assert len(prediction_batch) == len(sources_batch)
            num_predicted += len(prediction_batch)
            for id, [prediction, sources] in enumerate(zip(prediction_batch, sources_batch)):
                target = target_list[batch_id * predict_batch_size + id]
                writer.write(f'{sep_str.join(sources)}\t{prediction}\t{target}\n')
            if batch_id % 5 == 0:
                cost_time = (time.time() - start_time) / 60.0
                print("%s batch_id=%d/%d, predict %d/%d examples, cost %.2fmin." %
                      (curLine(), batch_id + 1, batch_num, num_predicted, number, cost_time))
    cost_time = (time.time() - start_time) / 60.0
    print(
        f'{curLine()} {num_predicted} predictions saved to:{FLAGS.output_file}, cost {cost_time} min, ave {cost_time/num_predicted*60} s.')

    data = []
    data.append({"question": "文件柜费用。",
              "context": "档案内阁平均费用.。办公室档案柜的价格取决于材料的质量、抽屉的数量、橱柜的尺寸以及锁紧系统的类型。两个抽屉的垂直文件通常要花20到150美元,而五个抽屉的垂直文件可以花250到1000美元购买。",
              "answer": "两个抽屉的垂直文件通常要花20到150美元,而五个抽屉的垂直文件可以花250到1000美元购买。"})
    data.append({"question": "在大蹦床上跳,能燃烧多少卡路里？",
              "context": "蹦床上的跳跃消耗的卡路里仅是跑步、游泳和骑自行车等更有力的运动的一小部分,但一次锻炼仍能消耗数卡路里。根据卡洛里实验室的说法,一个125磅重的人在蹦床上跳一小时就能燃烧143卡路里。一个体重185磅的人会用同样的运动消耗210卡路里。",
              "answer": "一个125磅重的人一小时能燃烧143卡路里。"})
    data.append({"question":"十八大以来思想文化建设取得哪些重大进展",
                 "context": "加强党对意识形态工作的领导，党的理论创新全面推进，马克思主义在意识形态领域的指导地位更加鲜明，中国特色社会主义和中国梦深入人心，社会主义核心价值观和中华优秀传统文化广泛弘扬，群众性精神文明创建活动扎实开展。公共文化服务水平不断提高，文艺创作持续繁荣，文化事业和文化产业蓬勃发展，互联网建设管理运用不断完善，全民健身和竞技体育全面发展。主旋律更加响亮，正能量更加强劲，文化自信得到彰显，国家文化软实力和中华文化影响力大幅提升，全党全社会思想上的团结统一更加巩固。",
                 "answer": "共文化服务水平不断提高，文艺创作持续繁荣，文化事业和文化产业蓬勃发展，互联网建设管理运用不断完善，全民健身和竞技体育全面发展。"})

    data.append({"question":"心室扑动是指什么?",
                 "context": "心室扑动简称为室扑，是指心室肌快而微弱的收缩或不协调的快速扑动。结果是心脏无排血，心音和脉搏消失，心、脑等器官和周围组织血液灌注停止，阿-斯综合征发作和猝死。心电图特点：①P波消失，出现连续和比较规则的大振幅波，呈正弦图形，频率约250次/分，不能区分QRS波群和ST-T波段；② 持续时间较短，常于数秒或数分钟内转变为室速或室颤。常见于急性心肌梗死等严重器质性心脏病患者的室性心律失常。",
                 "answer": "心室扑动简称为室扑，是指心室肌快而微弱的收缩或不协调的快速扑动。"})
    data.append({"question":"酒精性脑萎缩是指什么",
                 "context": "酒精性脑萎缩是指性酒精消耗所致的大脑组织不可逆的减少。多见于有长期大量饮酒史的中老年男性，一般均有慢性酒精中毒的表现。发病隐袭，且逐渐缓慢进展，主要特点是早期常有焦虑不安、头痛、失眠、乏力等，逐渐出现智力衰退和人格改变。除非有严重脑萎缩，一般无明显痴呆。可较长时期保持良好的工作能力。",
                 "answer": "酒精性脑萎缩是指性酒精消耗所致的大脑组织不可逆的减少。"})

    data.append({"question":"新型冠状病毒全球研究",
                 "context": "世界卫生组织和“全球传染病防控研究合作组织”11日在瑞士日内瓦共同举办“科研路线图:新型冠状病毒全球研究与创新论坛”。世卫组织总干事谭德塞在论坛上发表开幕讲话,强调论坛的目标之一是达成一份“科研路线图”",
                 "answer": "世界卫生组织和全球传染病防控研究合作组织11日在瑞士日内瓦共同举办科研路线图:新型冠状病毒全球研究与创新论坛"})

    data.append({"question":"月明寺",
                 "context": "月明寺位于满城县大册营镇岗头村,始建于唐朝中期,是一座历史悠久,得到高僧辈出,名播海内外的千年古刹。遗留下来的佛舍利双塔就是历史的见证,久传不衰的有关月明寺故事,如:《观音送粮》、《地藏菩萨施水》、《竹篮取水》、《张柔悟禅》……延续了一代又一代人。岗头村曾是水泽之地,鱼米之乡,是一个人杰地灵,民风淳朴的地方,北傍绵延千里泉水叮咚,松涛阵阵的太行山麓",
                 "answer": "月明寺位于满城县，是一座历史悠久,得到高僧辈出,名播海内外的千年古刹"})
    data.append({"question":"清西陵行宫",
                 "context": "清西陵行宫始建于乾隆十三年三月(1748年),完工于同年八月,是乾隆皇帝专为拜谒其父雍正的泰陵而建的。与其同时还建有房山的黄新庄行宫、涿州的丰壁店行宫、涞水的秋澜行宫。随着岁月的流逝,这三座行宫已荡然无存,而梁格庄行宫(清西陵行宫)却比较完好地保留下来。清西陵行宫,坐落于梁格庄村西,与御用喇嘛庙永福寺毗邻。南面有座小山叫龟山,北易水河沿山脚潺潺流过,山清水秀,风景宜人",
                 "answer": "清西陵行宫始建于乾隆十三年三月,完工于同年八月,是乾隆皇帝专为拜谒其父雍正的泰陵而建的。"})
    data.append({"question":"苏州园林",
                 "context": "俗语说:“江南园林甲全国, 苏州园林甲江南。”苏州古典园林, 简称苏州园林, 是世界文化遗产、国家aaaaa级旅游景区, 中国十大风景名胜之一, 素有“园林之城”之称, 被誉为“咫尺之内再造乾坤”, 它始于春秋时期吴国建都姑苏时, 形成于五代, 成熟于宋代, 兴旺鼎盛于明清, 现保存完整的有60多处, 对外开放的有19处, 主要有沧浪亭、狮子林、拙政园、留园、网师园、怡园等园林",
                 "answer": "苏州古典园林, 是世界文化遗产、国家aaaaa级旅游景区, 中国十大风景名胜之一"})

    start_time = time.time()
    sources_batch = []
    for corpus in data:
        sources_batch.append((corpus["question"],corpus["context"]))
    prediction_batch = predictor.predict_batch(sources_batch=sources_batch)
    assert len(prediction_batch) == len(sources_batch)
    for id, [prediction, sources] in enumerate(zip(prediction_batch, sources_batch)):
        print(curLine(), id, "prediction:", prediction)
    cost_time = (time.time() - start_time) / 1.0
    print(curLine(), "cost %f s" % cost_time)

    ###  摘要生成失败，请人工干预

if __name__ == '__main__':
    app.run(main)
