# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/11/2020 10:57 AM

from typing import *

import re
import cv2
import json
import string
from pathlib import Path

from torchtext.data import Field, RawField
import numpy as np

from utils.entities_list import Entities_list
from utils.class_utils import keys_vocab_cls, iob_labels_vocab_cls, entities_vocab_cls

MAX_BOXES_NUM = 70  # limit max number boxes of every documents
MAX_TRANSCRIPT_LEN = 50  # limit max length text of every box

# text string label converter
TextSegmentsField = Field(sequential=True, use_vocab=True, include_lengths=True, batch_first=True)
TextSegmentsField.vocab = keys_vocab_cls
# iob string label converter
IOBTagsField = Field(sequential=True, is_target=True, use_vocab=True, batch_first=True)
IOBTagsField.vocab = iob_labels_vocab_cls


class Document:
    def __init__(self, boxes_and_transcripts_file: Path, image_file: Path,
                 resized_image_size: Tuple[int, int] = (480, 960),
                 iob_tagging_type: str = 'box_level', entities_file: Path = None, training: bool = True,
                 image_index=None):
        '''
        An item returned by dataset.

        :param boxes_and_transcripts_file: gt or ocr results file
        :param image_file: whole images file
        :param resized_image_size: resize whole image size, (w, h)
        :param iob_tagging_type: 'box_level', 'document_level', 'box_and_within_box_level'
        :param entities_file: exactly entity type and entity value of documents, json file
        :param training: True for train and validation mode, False for test mode. True will also load labels,
        and entities_file must be set.
        :param image_index: image index, used to get image file name
        '''
        self.resized_image_size = resized_image_size
        self.training = training
        assert iob_tagging_type in ['box_level', 'document_level', 'box_and_within_box_level'], \
            'iob tagging type {} is not supported'.format(iob_tagging_type)
        self.iob_tagging_type = iob_tagging_type

        # For easier debug:
        # we will know what we are running on.
        self.image_filename = image_file.as_posix()

        try:
            # read boxes, transcripts, and entity types of boxes in one documents from boxes_and_transcripts file
            # match with regex pattern: index,x1,y1,x2,y2,x3,y3,x4,y4,transcript,type from boxes_and_transcripts tsv file
            # data format as [(index, points, transcription, entity_type)...]
            if self.training:
                # boxes_and_transcripts_data = [(index, [x1, y1, ...], transcript, entity_type), ...]
                boxes_and_transcripts_data = read_gt_file_with_box_entity_type(boxes_and_transcripts_file.as_posix())
            else:
                boxes_and_transcripts_data = read_ocr_file_without_box_entity_type(
                    boxes_and_transcripts_file.as_posix())

            # Sort the box based on the position.
            boxes_and_transcripts_data = sort_box_with_list(boxes_and_transcripts_data)

            # read image
            image = cv2.imread(image_file.as_posix())
        except Exception as e:
            raise IOError('Error occurs in image {}: {}'.format(image_file.stem, e.args))

        boxes, transcripts, box_entity_types = [], [], []
        if self.training:
            for index, points, transcript, entity_type in boxes_and_transcripts_data:
                if len(transcript) == 0:
                    transcript = ' '
                boxes.append(points)
                transcripts.append(transcript)
                box_entity_types.append(entity_type)
        else:
            for index, points, transcript in boxes_and_transcripts_data:
                if len(transcript) == 0:
                    transcript = ' '
                boxes.append(points)
                transcripts.append(transcript)

        # Limit the number of boxes and number of transcripts to process.
        boxes_num = min(len(boxes), MAX_BOXES_NUM)
        transcript_len = min(max([len(t) for t in transcripts[:boxes_num]]), MAX_TRANSCRIPT_LEN)
        mask = np.zeros((boxes_num, transcript_len), dtype=int)

        relation_features = np.zeros((boxes_num, boxes_num, 6))

        try:

            height, width, _ = image.shape

            # resize image
            image = cv2.resize(image, self.resized_image_size, interpolation=cv2.INTER_LINEAR)
            x_scale = self.resized_image_size[0] / width
            y_scale = self.resized_image_size[1] / height

            # get min area box for each (original) boxes, for calculate initial relation features
            min_area_boxes = [cv2.minAreaRect(np.array(box, dtype=np.float32).reshape(4, 2)) for box in
                              boxes[:boxes_num]]

            # calculate resized image box coordinate, and initial relation features between boxes (nodes)
            resized_boxes = []
            for i in range(boxes_num):
                box_i = boxes[i]
                transcript_i = transcripts[i]

                # get resized images's boxes coordinate, used to ROIAlign in Encoder layer
                resized_box_i = [int(np.round(pos * x_scale)) if i % 2 == 0 else int(np.round(pos * y_scale))
                                 for i, pos in enumerate(box_i)]

                resized_rect_output_i = cv2.minAreaRect(np.array(resized_box_i, dtype=np.float32).reshape(4, 2))
                resized_box_i = cv2.boxPoints(resized_rect_output_i)
                resized_box_i = resized_box_i.reshape((8,))
                resized_boxes.append(resized_box_i)

                # enumerate each box, calculate relation features between i and other nodes.
                # formula (9)
                self.relation_features_between_ij_nodes(boxes_num, i, min_area_boxes, relation_features, transcript_i,
                                                        transcripts)

            relation_features = normalize_relation_features(relation_features, width=width, height=height)
            # The length of texts of each segment.
            text_segments = [list(trans) for trans in transcripts[:boxes_num]]

            if self.training:
                # assign iob label to input text through exactly match way, this process needs entity-level label
                if self.iob_tagging_type != 'box_level':
                    with entities_file.open() as f:
                        entities = json.load(f)

                if self.iob_tagging_type == 'box_level':
                    # convert transcript of every boxes to iob label, using entity type of corresponding box
                    iob_tags_label = text2iob_label_with_box_level_match(box_entity_types[:boxes_num],
                                                                         transcripts[:boxes_num])
                elif self.iob_tagging_type == 'document_level':
                    # convert transcripts to iob label using document level tagging match method, all transcripts will
                    # be concatenated as a sequences
                    iob_tags_label = text2iob_label_with_document_level_exactly_match(transcripts[:boxes_num], entities)

                elif self.iob_tagging_type == 'box_and_within_box_level':
                    # perform exactly tagging within specific box, box_level_entities parames will perform boex level tagging.
                    iob_tags_label = text2iob_label_with_box_and_within_box_exactly_level(box_entity_types[:boxes_num],
                                                                                          transcripts[:boxes_num],
                                                                                          entities, ['address'])

                iob_tags_label = IOBTagsField.process(iob_tags_label)[:, :transcript_len].numpy()
                box_entity_types = [entities_vocab_cls.stoi[t] for t in box_entity_types[:boxes_num]]

            # texts shape is (num_texts, max_texts_len), texts_len shape is (num_texts,)
            texts, texts_len = TextSegmentsField.process(text_segments)
            texts = texts[:, :transcript_len].numpy()
            texts_len = np.clip(texts_len.numpy(), 0, transcript_len)
            text_segments = (texts, texts_len)

            for i in range(boxes_num):
                mask[i, :texts_len[i]] = 1

            self.whole_image = RawField().preprocess(image)
            self.text_segments = TextSegmentsField.preprocess(text_segments)  # (text, texts_len)
            self.boxes_coordinate = RawField().preprocess(resized_boxes)
            self.relation_features = RawField().preprocess(relation_features)
            self.mask = RawField().preprocess(mask)
            self.boxes_num = RawField().preprocess(boxes_num)
            self.transcript_len = RawField().preprocess(transcript_len)  # max transcript len of current document
            if self.training:
                self.iob_tags_label = IOBTagsField.preprocess(iob_tags_label)
            else:
                self.image_index = RawField().preprocess(image_index)

        except Exception as e:
            raise RuntimeError('Error occurs in image {}: {}'.format(boxes_and_transcripts_file.stem, e.args))

    def relation_features_between_ij_nodes(self, boxes_num, i, min_area_boxes, relation_features, transcript_i,
                                           transcripts):
        '''
        calculate node i and other nodes' initial relation features.
        :param boxes_num:
        :param i:
        :param min_area_boxes: the min rectangle of (original) points.
        :param relation_features: np.array, boxes_num x boxes_num x 6
        :param transcript_i:  transcripts[i]
        :param transcripts:
        :return:
        '''
        for j in range(boxes_num):
            transcript_j = transcripts[j]

            rect_output_i = min_area_boxes[i]
            rect_output_j = min_area_boxes[j]

            # Centers of rect_of_box_i and rect_of_box_j.
            center_i = rect_output_i[0]
            center_j = rect_output_j[0]

            width_i, height_i = rect_output_i[1]
            width_j, height_j = rect_output_j[1]

            # Center distances of boxes on x-axis.
            relation_features[i, j, 0] = np.abs(center_i[0] - center_j[0]) \
                if np.abs(center_i[0] - center_j[0]) is not None else -1  # x_ij

            # Center distances of boxes on y-axis.
            relation_features[i, j, 1] = np.abs(center_i[1] - center_j[1]) \
                if np.abs(center_i[1] - center_j[1]) is not None else -1  # y_ij

            relation_features[i, j, 2] = width_i / (height_i) \
                if height_i != 0 and width_i / (height_i) is not None else -1  # w_i/h_i

            relation_features[i, j, 3] = height_j / (height_i) \
                if height_i != 0 and height_j / (height_i) is not None else -1  # h_j/h_i

            relation_features[i, j, 4] = width_j / (height_i) \
                if height_i != 0 and width_j / (height_i) is not None else -1  # w_j/h_i

            relation_features[i, j, 5] = len(transcript_j) / (len(transcript_i)) \
                if len(transcript_j) / (len(transcript_i)) is not None else -1  # T_j/T_i


def read_gt_file_with_box_entity_type(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        document_text = f.read()

    # match pattern in document: index,x1,y1,x2,y2,x3,y3,x4,y4,transcript,box_entity_type
    regex = r"^\s*(-?\d+)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*," \
            r"\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,(.*),(.*)\n?$"

    matches = re.finditer(regex, document_text, re.MULTILINE)

    res = []
    for matchNum, match in enumerate(matches, start=1):
        index = int(match.group(1))
        points = [float(match.group(i)) for i in range(2, 10)]
        transcription = str(match.group(10))
        entity_type = str(match.group(11))
        res.append((index, points, transcription, entity_type))
    return res


def read_ocr_file_without_box_entity_type(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        document_text = f.read()

    # match pattern in document: index,x1,y1,x2,y2,x3,y3,x4,y4,transcript
    regex = r"^\s*(-?\d+)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*," \
            r"\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,(.*)\n?$"

    matches = re.finditer(regex, document_text, re.MULTILINE)

    res = []
    for matchNum, match in enumerate(matches, start=1):
        index = int(match.group(1))
        points = [float(match.group(i)) for i in range(2, 10)]
        transcription = str(match.group(10))
        res.append((index, points, transcription))
    return res


def sort_box_with_list(data: List[Tuple], left_right_first=False):
    def compare_key(x):
        #  x is (index, points, transcription, type) or (index, points, transcription)
        points = x[1]
        box = np.array([[points[0], points[1]], [points[2], points[3]], [points[4], points[5]], [points[6], points[7]]],
                       dtype=np.float32)
        rect = cv2.minAreaRect(box)
        center = rect[0]
        if left_right_first:
            return center[0], center[1]
        else:
            return center[1], center[0]

    data = sorted(data, key=compare_key)
    return data


def normalize_relation_features(feat: np.ndarray, width: int, height: int):
    # TODO check _NoValueType value (None)
    np.clip(feat, 1e-8, np.inf)
    feat[:, :, 0] = feat[:, :, 0] / width
    feat[:, :, 1] = feat[:, :, 1] / height

    # The second graph to the 6th graph.
    for i in range(2, 6):
        feat_ij = feat[:, :, i]
        max_value = np.max(feat_ij)
        min_value = np.min(feat_ij)
        if max_value != min_value:
            feat[:, :, i] = feat[:, :, i] - min_value / (max_value - min_value)
    return feat


def text2iob_label_with_box_level_match(annotation_box_types: List[str], transcripts: List[str]) -> List[List[str]]:
    '''
     convert transcripts to iob label using box level tagging match method
    :param annotation_box_types: each transcripts box belongs to the corresponding entity types
    :param transcripts: transcripts of documents
    :return:
    '''
    tags = []
    for entity_type, transcript in zip(annotation_box_types, transcripts):
        if entity_type in Entities_list:
            if len(transcript) == 1:
                tags.append(['B-{}'.format(entity_type)])
            else:
                tag = ['I-{}'.format(entity_type)] * len(transcript)
                tag[0] = 'B-{}'.format(entity_type)
                tags.append(tag)
        else:
            tags.append(['O'] * len(transcript))

    return tags


def text2iob_label_with_document_level_exactly_match(transcripts: List[str], exactly_entities_label: Dict) -> List[
    List[str]]:
    '''
     convert transcripts to iob label using document level tagging match method,
     all transcripts will be concatenated as a sequences.
    :param transcripts: transcripts of documents
    :param exactly_entities_label: exactly entity type and entity value of documents
    :return:
    '''
    concatenated_sequences = []
    sequences_len = []
    for transcript in transcripts:
        concatenated_sequences.extend(list(transcript))
        sequences_len.append(len(transcript))

    result_tags = ['O'] * len(concatenated_sequences)
    for entity_type, entity_value in exactly_entities_label.items():
        if entity_type not in Entities_list:
            continue
        (src_seq, src_idx), (tgt_seq, _) = preprocess_transcripts(concatenated_sequences), preprocess_transcripts(
            entity_value)
        src_len, tgt_len = len(src_seq), len(tgt_seq)
        if tgt_len == 0:
            continue

        for i in range(src_len - tgt_len + 1):
            if src_seq[i:i + tgt_len] == tgt_seq:
                tag = ['I-{}'.format(entity_type)] * (src_idx[i + tgt_len - 1] - src_idx[i] + 1)
                tag[0] = 'B-{}'.format(entity_type)
                result_tags[src_idx[i]:src_idx[i + tgt_len - 1] + 1] = tag

    tagged_transcript = []
    start = 0
    for length in sequences_len:
        tagged_transcript.append(result_tags[start: start + length])
        start = start + length
        if start >= len(result_tags):
            break
    return tagged_transcript


def text2iob_label_with_box_and_within_box_exactly_level(annotation_box_types: List[str],
                                                         transcripts: List[str],
                                                         exactly_entities_label: Dict[str, str],
                                                         box_level_entities: List[str]) -> List[List[str]]:
    '''
     box_level_entities will perform box level tagging, others will perform exactly matching within specific box.
    :param annotation_box_types: each transcripts box belongs to the corresponding entity types
    :param transcripts: transcripts of documents
    :param exactly_entities_label: exactly entity type and entity value of documents
    :param box_level_entities: using box level label tagging, this result is same as
                    function of text2iob_label_with_box_level_match
    :return:
    '''

    def exactly_match_within_box(transcript: str, entity_type: str, entity_exactly_value: str):
        '''
        perform exactly match in the scope of current box
        :param transcript: the transcript of current box
        :param entity_type: the entity type of current box
        :param entity_exactly_value: exactly label value of corresponding entity type
        :return:
        '''
        matched = False

        # Preprocess remove the punctuations and whitespaces.
        (src_seq, src_idx), (tgt_seq, _) = preprocess_transcripts(transcript), preprocess_transcripts(
            entity_exactly_value)
        src_len, tgt_len = len(src_seq), len(tgt_seq)
        if tgt_len == 0:
            return matched, None

        result_tags = ['O'] * len(transcript)
        for i in range(src_len - tgt_len + 1):
            if src_seq[i:i + tgt_len] == tgt_seq:
                matched = True
                tag = ['I-{}'.format(entity_type)] * (src_idx[i + tgt_len - 1] - src_idx[i] + 1)
                tag[0] = 'B-{}'.format(entity_type)
                result_tags[src_idx[i]:src_idx[i + tgt_len - 1] + 1] = tag
                break

        return matched, result_tags

    tags = []
    for entity_type, transcript in zip(annotation_box_types, transcripts):
        entity_type = entity_type.strip()
        if entity_type in Entities_list:

            matched, resulted_tag = False, None
            if entity_type not in box_level_entities:
                matched, resulted_tag = exactly_match_within_box(transcript, entity_type,
                                                                 exactly_entities_label[entity_type])

            if matched:
                tags.append(resulted_tag)
            else:
                tag = ['I-{}'.format(entity_type)] * len(transcript)
                tag[0] = 'B-{}'.format(entity_type)
                tags.append(tag)
        else:
            tags.append(['O'] * len(transcript))

    return tags


def preprocess_transcripts(transcripts: List[str]):
    '''
    preprocess texts into separated word-level list, this is helpful to matching tagging label between source and target label,
    e.g. source: xxxx hello ! world xxxx  target: xxxx hello world xxxx,
    we want to match 'hello ! world' with 'hello world' to decrease the impact of ocr bad result.
    :param transcripts:
    :return: seq: the cleaned sequence, idx: the corresponding indices.
    '''
    seq, idx = [], []
    for index, x in enumerate(transcripts):
        if x not in string.punctuation and x not in string.whitespace:
            seq.append(x)
            idx.append(index)
    return seq, idx
