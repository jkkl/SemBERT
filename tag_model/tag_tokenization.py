# coding=utf-8

from __future__ import absolute_import
from __future__ import division
import collections
import pickle
import os
import logging
from os.path import join, exists
from os import makedirs


logger = logging.getLogger(__name__)
# origin 
# TAG_VOCAB = ['[PAD]','[CLS]', '[SEP]', 'B-V', 'I-V', 'B-ARG0', 'I-ARG0', 'B-ARG1', 'I-ARG1', 'B-ARG2', 'I-ARG2', 'B-ARG4', 'I-ARG4', 'B-ARGM-TMP', 'I-ARGM-TMP', 'B-ARGM-LOC', 'I-ARGM-LOC', 'B-ARGM-CAU', 'I-ARGM-CAU', 'B-ARGM-PRP', 'I-ARGM-PRP', 'O']
#or load the full vocab of SRL, this will not affect the performance too much.
# hanlp tag 
# TAG_VOCAB = ['[PAD]', '[CLS]', '[SEP]', 'B-PRED','B-ARGM-CRD', 'I-ARGM-CRD','B-ARGM-QTY', 'I-ARGM-QTY', 'B-ARGM', 'B-ARGM-ADV', 'B-ARG1', 'B-ARGM-DIS', 'B-ARG0', 'I-ARG1', 'I-ARGM-ADV', 'I-PRED', 'I-ARG0', 'B-Sup', 'B-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-DIS', 'B-ARG2', 'I-ARG2', 'B-ARGM-LOC', 'I-ARGM-LOC', 'B-ARGM-EXT', 'I-ARGM-EXT', 'B-ARGM-BNF', 'I-ARGM-BNF', 'B-C-ARGM-ADV', 'B-ARGM-MNR', 'I-ARGM-MNR', 'B-ARGM-TPC', 'I-ARGM-TPC', 'B-ARG3', 'B-ARGM-FRQ', 'I-ARGM-FRQ', 'B-ARG1-PRD', 'I-ARG1-PRD', 'B-ARGM-PRP', 'I-ARGM-PRP', 'B-ARG0-PRD', 'B-C-ARGM-MNR', 'I-C-ARGM-MNR', 'B-ARGM-CND', 'B-ARGM-DIR', 'I-ARGM-DIR', 'B-C-ARGM-TMP', 'B-ARG2-PRD', 'B-ARG0-CRD', 'I-ARG0-CRD', 'B-R-ARGM-MNR', 'B-C-ARGM-PRP', 'B-C-ARGM-LOC', 'I-C-ARGM-LOC', 'B-R-ARG0', 'B-ARG1-TPC', 'I-ARG1-TPC', 'B-ARG0-PSR', 'I-ARG0-PSR', 'I-R-ARG0', 'I-ARG0-PRD', 'B-ARGM-PRD', 'B-C-ARG3', 'I-C-ARG3', 'B-R-ARGM-ADV', 'B-ARG0-PSE', 'I-R-ARGM-ADV', 'B-ARG2-CRD', 'I-ARG2-CRD', 'I-ARG3', 'I-ARG2-PSR', 'B-ARG1-PSE', 'I-ARG1-PSE', 'B-ARGM-NEG', 'B-C-ARGM-TPC', 'B-ARG1-CRD', 'I-ARGM-CND', 'I-C-ARGM-TPC', 'B-C-ARG2', 'B-ARG2-PSE', 'I-Sup', 'B-ARG2-PSR', 'B-ARGM-DGR', 'I-ARGM-NEG', 'B-ARG2-QTY', 'I-ARGM-DGR', 'I-ARG2-QTY', 'B-C-ARG1', 'I-C-ARG1', 'B-R-ARGM-LOC', 'B-ARG1-QTY', 'I-ARG1-QTY', 'B-ARG1-PSR', 'B-ARG0-ADV', 'I-ARG1-CRD', 'B-rel-EXT', 'I-rel-EXT', 'I-ARG0-PSE', 'O']
# TAG_VOCAB = ['[PAD]', '[CLS]', '[SEP]','O', 'I-ARG1', 'B-PRED', 'B-ARG0', 'B-ARG1', 'I-ARG0', 'B-ARGM-ADV', 'I-PRED', 'I-ARGM-ADV', 'I-ARGM-TMP', 'I-ARG2', 'B-ARGM-TMP', 'B-ARGM-DIS', 'I-ARGM-DIS', 'B-ARG2']
TAG_VOCAB = ['[PAD]', '[CLS]', '[SEP]','O', 'I-ARG1', 'B-PRED', 'B-ARG0', 'B-ARG1', 'I-ARG0', 'I-PRED', 'I-ARG2', 'B-ARG2']

def load_tag_vocab(tag_vocab_file):
    vocab_list = ["[PAD]", "[CLS]", "[SEP]"]
    with open(tag_vocab_file, 'rb') as f:
        vocab_list.extend(pickle.load(f))
    return vocab_list


class TagTokenizer(object):
    def __init__(self):
        self.tag_vocab = TAG_VOCAB 
        self.ids_to_tags = collections.OrderedDict(
            [(ids, tag) for ids, tag in enumerate(self.tag_vocab)])

    def convert_tags_to_ids(self, tags):
        """Converts a sequence of tags into ids using the vocab."""
        ids = []
        for tag in tags:
            if tag not in self.tag_vocab:
                tag = 'O'
            ids.append(self.tag_vocab.index(tag))

        return ids

    def convert_ids_to_tags(self, ids):
        """Converts a sequence of ids into tags using the vocab."""
        tags = []
        for i in ids:
            tags.append(self.ids_to_tags[i])
        return tags
