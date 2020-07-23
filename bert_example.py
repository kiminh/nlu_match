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

"""Build BERT Examples from text (source, target) pairs."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import tensorflow as tf
from bert import tokenization

from curLine_file import curLine


class my_tokenizer_class(object):
    def __init__(self, vocab_file, do_lower_case):
        self.full_tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=do_lower_case)
    # 需要包装一下，因为如果直接对中文用full_tokenizer.tokenize，会忽略文本中的空格
    def tokenize(self, text):
        segments = text.split(" ")
        word_pieces = []
        for segId, segment in enumerate(segments):
            if segId > 0:
                word_pieces.append(" ")
            word_pieces.extend(self.full_tokenizer.tokenize(segment))
        return word_pieces
    def convert_tokens_to_ids(self, tokens):
        id_list = [self.full_tokenizer.vocab[t]
                   if t != " " else self.full_tokenizer.vocab["[unused20]"] for t in tokens]
        return id_list

class BertExample(object):
  """Class for training and inference examples for BERT.

  Attributes:
    editing_task: The EditingTask from which this example was created. Needed
      when realizing labels predicted for this example.
    features: Feature dictionary.
  """

  def __init__(self, input_ids,
               input_mask,
               segment_ids,
               labels):
    input_len = len(input_ids)
    if not (input_len == len(input_mask) and input_len == len(segment_ids)):
      raise ValueError(
          'All feature lists should have the same length ({})'.format(
              input_len))

    self.features = collections.OrderedDict([
        ('input_ids', input_ids),
        ('input_mask', input_mask),
        ('segment_ids', segment_ids),
        ('labels', labels),
    ])

  def pad_to_max_length(self, max_seq_length, pad_token_id):
    """Pad the feature vectors so that they all have max_seq_length.

    Args:
      max_seq_length: The length that features will have after padding.
      pad_token_id: input_ids feature is padded with this ID, other features
        with ID 0.
    """
    pad_len = max_seq_length - len(self.features['input_ids'])
    for key in self.features:
      if key == "labels":
        continue
      pad_id = pad_token_id if key == 'input_ids' else 0
      self.features[key].extend([pad_id] * pad_len)
      if len(self.features[key]) != max_seq_length:
        raise ValueError('{} has length {} (should be {}).'.format(
            key, len(self.features[key]), max_seq_length))

  def to_tf_example(self):
    """Returns this object as a tf.Example."""

    def int_feature(values):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    tf_features = collections.OrderedDict([
        (key, int_feature(val)) if type(val) is not int else (key, int_feature([val])) for key, val in self.features.items()
    ])
    return tf.train.Example(features=tf.train.Features(feature=tf_features))



class BertExampleBuilder(object):
  """Builder class for BertExample objects."""

  def __init__(self, label_map, vocab_file,
               max_seq_length, do_lower_case):
    """Initializes an instance of BertExampleBuilder.

    Args:
      label_map: Mapping from tags to tag IDs.
      vocab_file: Path to BERT vocabulary file.
      max_seq_length: Maximum sequence length.
      do_lower_case: Whether to lower case the input text. Should be True for
        uncased models and False for cased models.
    """
    self._label_map = label_map
    self._tokenizer = my_tokenizer_class(vocab_file, do_lower_case=do_lower_case)
    self._max_seq_length = max_seq_length
    self._pad_id = self._get_pad_id()


  def build_bert_example(
      self,
      sources,
      target = None,
      location = None
  ):
    """Constructs a BERT Example.

    Args:
      sources: List of source texts.

    Returns:
      BertExample, or None if the conversion from text to tags was infeasible
      and use_arbitrary_target_ids_for_infeasible_examples == False.
    """
    # Compute target labels.
    sep_mark = '[SEP]'
    source_tokens = self._tokenizer.tokenize(sources[-1]) #current query
    if len(sources) > 1:
      source_tokens = self._tokenizer.tokenize(sources[-2]) + ['[SEP]'] + source_tokens
      if len(sources) > 2:  # TODO context
          source_tokens = self._tokenizer.tokenize(sources[-3]) + ['[SEP]'] + source_tokens
    if target not in self._label_map:
      self._label_map[target] = len(self._label_map)
    labels = self._label_map[target]

    #  截断到self._max_seq_length - 2
    tokens = self._truncate_list(source_tokens)
    # if len(source_tokens)>self._max_seq_length - 2:
    #   print(curLine(), "%d tokens is to long," % len(source_tokens), "truncate task.source_tokens:", source_tokens)
    #   print(curLine(), len(tokens), "tokens:", tokens, "\n")
    input_tokens = ['[CLS]'] + tokens + [sep_mark]

    context_len_list = [] # i+1 for i,t in enumerate(tokens) if t==sep_mark]
    for i, t in enumerate(tokens):
        if t == sep_mark:
            if len(context_len_list) > 0:
                context_len_list.append(i+1-context_len_list[-1])
            else:
                context_len_list.append(i+1)
    segment_ids = [0]
    segment_index = 0
    context_len = 0
    for context_len in context_len_list:
        segment_ids.extend([segment_index] * context_len) #
        segment_index += 1
    segment_ids.extend([segment_index] * (len(tokens) - len(segment_ids) + 2))
    assert len(segment_ids) == len(input_tokens)  # TODO

    input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)

    example = BertExample(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        labels=labels)
    example.pad_to_max_length(self._max_seq_length, self._pad_id)
    return example, input_tokens

  def _truncate_list(self, x): # 从后截断
    """Returns truncated version of x according to the self._max_seq_length."""
    # Save two slots for the first [CLS] token and the last [SEP] token.
    return x[-(self._max_seq_length - 2):]

  def _get_pad_id(self):
    """Returns the ID of the [PAD] token (or 0 if it's not in the vocab)."""
    try:
      return self._tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
    except KeyError:
      return 0
