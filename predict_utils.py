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
"""Utility functions for running inference with a LaserTagger model."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
from collections import defaultdict
from curLine_file import curLine
import tagging


class LaserTaggerPredictor(object):
  """Class for computing and realizing predictions with LaserTagger."""

  def __init__(self, tf_predictor,
               example_builder,
               label_map):
    """Initializes an instance of LaserTaggerPredictor.

    Args:
      tf_predictor: Loaded Tensorflow model.
      example_builder: BERT example builder.
      label_map: Mapping from tags to tag IDs.
    """
    self._predictor = tf_predictor
    self._example_builder = example_builder
    self._id_2_tag = {
        tag_id: tag for tag, tag_id in label_map.items()
    }

  def predict_batch(self, sources_batch, location_batch=None):  # 由predict改成
    """Returns realized prediction for given sources."""
    # Predict tag IDs.
    keys = ['input_ids', 'input_mask', 'segment_ids']
    input_info = defaultdict(list)
    example_list = []
    location = None
    for id, sources in enumerate(sources_batch):
      if location_batch is not None:
        location = location_batch[id]  #  表示是否能修改
      example = self._example_builder.build_bert_example(sources, location=location)
      if example is None:
        raise ValueError("Example couldn't be built.")
      for key in keys:
        input_info[key].append(example.features[key])
      example_list.append(example)

    out = self._predictor(input_info)
    prediction_list= []
    for output in out['pred']:
      predicted_ids = output.tolist()
      prediction = self._id_2_tag[predicted_ids]
      prediction_list.append(prediction)
    return prediction_list
