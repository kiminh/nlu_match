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
"""Calculates evaluation scores for a prediction TSV file.

The prediction file is produced by predict_main.py and should contain 3 or more
columns:
  1: sources (concatenated)
  2: prediction
  3-n: targets (1 or more)
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import app, flags, logging
import score_lib
from bert import tokenization
from curLine_file import curLine

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'prediction_file', None,
    'TSV file containing source, prediction, and target columns.')
flags.DEFINE_bool(
    'do_lower_case', True,
    'Whether score computation should be case insensitive (in the LaserTagger '
    'paper this was set to True).')
flags.DEFINE_string('vocab_file', None, 'Path to the BERT vocabulary file.')

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  flags.mark_flag_as_required('prediction_file')

  sources, predictions, target_lists = score_lib.read_data(
      FLAGS.prediction_file, FLAGS.do_lower_case)
  logging.info(f'Read file: {FLAGS.prediction_file}')
  tokenizer = tokenization.FullTokenizer(FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  exact = score_lib.compute_exact_score(predictions, target_lists)
  sari, keep, addition, deletion, length_sum, length_max = score_lib.compute_sari_scores(
      sources, predictions, target_lists, tokenizer=tokenizer)
  print('Num=%d, Exact score=%.3f, SARI score=%.3f, KEEP score=%.3f, ADDITION score=%.3f, DELETION score=%.3f, length_sum=%.3f, length_max=%d'
    % (len(predictions), 100*exact, 100*sari, 100*keep, 100*addition, 100*deletion, length_sum, length_max))


if __name__ == '__main__':
  app.run(main)
