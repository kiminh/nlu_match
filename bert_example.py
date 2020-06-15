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
               labels,
               token_start_indices):
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
    self._token_start_indices = token_start_indices

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
    self._tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=do_lower_case)
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
      target: Target text or None when building an example during inference.
      use_arbitrary_target_ids_for_infeasible_examples: Whether to build an
        example with arbitrary target ids even if the target can't be obtained
        via tagging.

    Returns:
      BertExample, or None if the conversion from text to tags was infeasible
      and use_arbitrary_target_ids_for_infeasible_examples == False.
    """
    # Compute target labels.
    sep_mark = '[SEP]'
    source_tokens = self._tokenizer.tokenize(sources[-1]) #current query
    if len(sources) > 1:  # TODO context
      source_tokens = self._tokenizer.tokenize(sources[0]) + ['[SEP]'] + source_tokens
    # print(curLine(), len(source_tokens), "source_tokens:", source_tokens)
    if target not in self._label_map:
      self._label_map[target] = len(self._label_map)
    labels = self._label_map[target]
    # tokens, token_start_indices = self._split_to_wordpieces(source_tokens) # TODO
    token_start_indices = []
    for t in source_tokens:
      token_start_indices.append(len(token_start_indices))

    # if len(tokens)>self._max_seq_length - 2:
    #   print(curLine(), "%d tokens is to long," % len(task.source_tokens), "truncate task.source_tokens:", task.source_tokens)
    #  截断到self._max_seq_length - 2
    tokens = self._truncate_list(source_tokens)
    # print(curLine(), tokens)  # TODO TODO

    input_tokens = ['[CLS]'] + tokens + ['[SEP]']
    if sep_mark in tokens:
      context_len = 1 + tokens.index(sep_mark)
      segment_ids = [0] * (context_len+1) + [1] * (len(tokens) - context_len + 1)  # cls and sep
    else:
      segment_ids = [0] * len(input_tokens)
    assert len(segment_ids) == len(input_tokens) # TODO

    input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)

    example = BertExample(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        labels=labels,
        token_start_indices=token_start_indices)
    example.pad_to_max_length(self._max_seq_length, self._pad_id)
    return example

  def _split_to_wordpieces(self, tokens):
    """Splits tokens (and the labels accordingly) to WordPieces.

    Args:
      tokens: Tokens to be split.
      labels: Labels (one per token) to be split.

    Returns:
      3-tuple with the split tokens, split labels, and the indices of the
      WordPieces that start a token.
    """
    bert_tokens = []  # Original tokens split into wordpieces.
    # Index of each wordpiece that starts a new token.
    token_start_indices = []
    for i, token in enumerate(tokens):
      # '+ 1' is because bert_tokens will be prepended by [CLS] token later.
      token_start_indices.append(len(bert_tokens) + 1)
      if token != "[SEP]":
        pieces = self._tokenizer.tokenize(token)
      else:
        pieces = ["[SEP]"]
      bert_tokens.extend(pieces)
    return bert_tokens, token_start_indices

  def _truncate_list(self, x):
    """Returns truncated version of x according to the self._max_seq_length."""
    # Save two slots for the first [CLS] token and the last [SEP] token.
    return x[:self._max_seq_length - 2]

  def _get_pad_id(self):
    """Returns the ID of the [PAD] token (or 0 if it's not in the vocab)."""
    try:
      return self._tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
    except KeyError:
      return 0
