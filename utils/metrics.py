# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/11/2020 10:02 PM

from typing import *
import numpy as np
import pandas as pd

import torch
from .span_based_f1 import SpanBasedF1Measure
import tabulate


class MetricTracker:
    def __init__(self, *keys, writer=None):
        '''
        loss metric tracker
        :param keys:
        :param writer:
        '''
        self.writer = writer
        columns = ['total', 'counts', 'average']
        self._data = pd.DataFrame(np.zeros((len(keys), len(columns))), index=keys, columns=columns)
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class SpanBasedF1MetricTracker:
    '''
    mEF metrics tracker
    '''

    def __init__(self, vocab, **kwargs):
        metric = SpanBasedF1Measure(vocab=vocab, **kwargs)
        self._metric = metric
        self.reset()

    def update(self, class_probailites, tags, mask):
        self._metric(class_probailites, tags, mask.float())

    def result(self):
        metric = self._metric.get_metric()
        data_dict = {}
        for k, v in metric.items():
            entity = k.split('-')[-1]

            item = data_dict.get(entity, {})
            if 'mEF' in k:
                item['mEF'] = v
            elif 'mEP' in k:
                item['mEP'] = v
            elif 'mER' in k:
                item['mER'] = v
            else:
                item['mEA'] = v
            data_dict[entity] = item

        return data_dict

    def reset(self):
        self._metric.reset()

    @staticmethod
    def dict2str(data_dict: Dict):
        data_list = [['name', 'mEP', 'mER', 'mEF', 'mEA']]
        for k, v in data_dict.items():
            data_list.append([k, v['mEP'], v['mER'], v['mEF'], v['mEA']])
        return tabulate.tabulate(data_list, tablefmt='grid', headers='firstrow')
