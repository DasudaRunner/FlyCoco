# coding:utf-8
# datetime:2020/9/24 4:24 下午
# author: haibo

import cv2
import numpy as np

class ReadImage(object):
    def __init__(self,reader='cv2'):
        self.reader = reader

    def __call__(self, res):
        filename = res['img_info']['file_name']
        if self.reader=='cv2':
            img = cv2.imread(filename)
        else:
            raise NotImplementedError('not supported {}'.format(self.reader))
        res['img'] = img
        return res



    def  __call__(self, res):
        ann_info = res['ann_info']