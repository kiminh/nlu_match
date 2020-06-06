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
import csv
import utils
from bert import tokenization
from curLine_file import curLine, normal_transformer, other_tag

sep_str = '$'
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
    intent_list = []
    sources_list = []
    predict_batch_size = 64
    limit = predict_batch_size * 15 * 2 # 5184 #　10001 #
    with tf.gfile.GFile(FLAGS.input_file) as f:
        reader = csv.reader(f)
        session_list = []
        for row_id, line in enumerate(reader):
            # (sessionId, raw_query)
            (sessionId, raw_query, domain_intent, param) = line
            query = normal_transformer(raw_query)
            if domain_intent == other_tag:
                domain = other_tag
            else:
                domain, intent = domain_intent.split(".")
            if row_id == 0:
                continue
            sources = []
            if row_id > 1 and sessionId == session_list[row_id - 2][0]:
                sources.append(session_list[row_id - 2][1])  # last query
            sources.append(query)
            session_list.append((sessionId, query))
            sources_list.append(sources)
            domain_list.append(domain)
            intent_list.append(intent)
            if len(sources_list) >= limit:
                print(colored("%s stop reading at %d to save time" %(curLine(), limit), "red"))
                break

    number = len(sources_list)  # 总样本数
    predict_batch_size = min(predict_batch_size, number)
    batch_num = math.ceil(float(number) / predict_batch_size)
    start_time = time.time()
    num_predicted = 0
    with tf.gfile.Open(FLAGS.output_file, 'w') as writer:
        writer.write("\t".join(["sessionId", "query", "prediction", "domain", "intent"]) + "\n")
        for batch_id in range(batch_num):
            sources_batch = sources_list[batch_id * predict_batch_size: (batch_id + 1) * predict_batch_size]
            prediction_batch = predictor.predict_batch(sources_batch=sources_batch)
            assert len(prediction_batch) == len(sources_batch)
            num_predicted += len(prediction_batch)
            for id, [prediction, sources] in enumerate(zip(prediction_batch, sources_batch)):
                domain = domain_list[batch_id * predict_batch_size + id]
                intent = intent_list[batch_id * predict_batch_size + id]
                sessionId, query = session_list[batch_id * predict_batch_size + id]
                writer.write("\t".join([sessionId, query, prediction, domain, intent]) + "\n")
            if batch_id % 5 == 0:
                cost_time = (time.time() - start_time) / 60.0
                print("%s batch_id=%d/%d, predict %d/%d examples, cost %.2fmin." %
                      (curLine(), batch_id + 1, batch_num, num_predicted, number, cost_time))
    cost_time = (time.time() - start_time) / 60.0
    print(
        f'{curLine()} {num_predicted} predictions saved to:{FLAGS.output_file}, cost {cost_time} min, ave {cost_time/num_predicted*60} s.')


    cost_time = (time.time() - start_time) / 1.0
    print(curLine(), "cost %f s" % cost_time)

if __name__ == '__main__':
    app.run(main)
