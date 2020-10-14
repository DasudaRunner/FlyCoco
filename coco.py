# coding:utf-8
# datetime:2020/9/24 12:20 上午
# author: haibo

from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import torch.utils.data as torch_data
from utils import *
import os.path as osp
import copy
import numpy as np
import cv2
import random
'''
img info:
{
   'license': 4,
   'file_name': '000000397133.jpg',
   'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg',
   'height': 427,
   'width': 640,
   'date_captured': '2013-11-14 17:02:52',
   'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg',
   'id': 397133
}

ann info:
{
   'segmentation': [[224.24, 297.18, 228.29, 297.18, 234.91, 298.29, 243.0, 297.55, 249.25, 296.45, 252.19, 294.98, 256.61, 292.4, 254.4, 264.08, 251.83, 262.61, 241.53, 260.04, 235.27, 259.67, 230.49, 259.67, 233.44, 255.25, 237.48, 250.47, 237.85, 243.85, 237.11, 240.54, 234.17, 242.01, 228.65, 249.37, 224.24, 255.62, 220.93, 262.61, 218.36, 267.39, 217.62, 268.5, 218.72, 295.71, 225.34, 297.55]],
   'area': 1481.3806499999994,
   'iscrowd': 0,
   'image_id': 397133,
   'bbox': [217.62, 240.54, 38.99, 57.75],
   'category_id': 44,
   'id': 82445
}
'''

class COCODataset(torch_data.Dataset):

    def __init__(self,
                 data_root,
                 img_prefix,
                 ann_file,
                 pipline,
                 training=False,
                 filter_small_img=True):
        self.data_root = data_root
        self.img_prefix = osp.join(self.data_root, img_prefix)
        self.ann_file = osp.join(self.data_root, ann_file)
        self.training = training
        self.filter_small_img = filter_small_img

        self.load_annotations()
        # print('before filter:', len(self.img_info))
        if self.filter_small_img:
            self._filter_small_imgs()
        # print('after filter:', len(self.img_info))

        self.pipline = pipline

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        one_train_data = self.get_one_img(idx)
        aug_train_data = self.pipline(one_train_data)
        return aug_train_data

    def get_one_img(self, idx=None):
        if not idx:
            idx = random.randint(0,len(self.img_info)-1)
        assert idx < len(self.img_info)
        img_info = self.img_info[idx]
        ann_info = self.get_ann_by_id(img_info['id'])
        ann_info['masks'] = self._decodeMasks(img_info,ann_info)
        result = dict(img_info=img_info,ann_info=ann_info)
        return result

    def get_ann_by_id(self,_img_id):
        ann_dict = self.coco.getAnnIds(imgIds=[_img_id])
        ann_info = self.coco.loadAnns(ann_dict)
        return self._parse_ann_from_meta(ann_info)

    def load_annotations(self):
        self.coco = COCO(self.ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        self.img_info = self.coco.loadImgs(self.img_ids)
        for i in self.img_info:
            i['file_name'] = osp.join(self.img_prefix,i['file_name'])

    def _filter_small_imgs(self, min_size=32):
        new_img_info = []
        anns_with_gt = set([a['image_id'] for a in self.coco.anns.values()])
        for idx, _img_info in enumerate(self.img_info):
            if self.img_ids[idx] in anns_with_gt and min(_img_info['width'], _img_info['height']) > min_size:
                new_img_info.append(self.img_info[idx])
        self.img_info = copy.deepcopy(new_img_info)

    def _parse_ann_from_meta(self,ann_info):
        '''
        :param ann_info: {}
        :return:
        '''

        gt_bboxes = []
        gt_labels = []
        gt_masks = []
        gt_bboxex_iscrowd = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore',False):
                continue
            # bbox: [217.62, 240.54, 38.99, 57.75]
            x,y,w,h = ann['bbox']

            # 'area': 1481.3806499999994
            if ann['area'] <=0 or w < 1 or h <1:
                continue

            bbox = [x, y, x+w-1, y+h-1]
            if ann.get('iscrowd',False):
                gt_bboxex_iscrowd.append(bbox)
            else:
                # 当前图片的所有box，及对应的mask和label
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                # 将 segmentation decode 为mask
                gt_masks.append(ann['segmentation'])

        # 此时gt_bboxes gt_labels gt_masks 都有可能为空
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes,dtype=np.float32)
            gt_labels = np.array(gt_labels,dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0,4),dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxex_iscrowd:
            gt_bboxex_iscrowd = np.array(gt_bboxex_iscrowd,dtype=np.float32)
        else:
            gt_bboxex_iscrowd = np.zeros((0, 4), dtype=np.float32)

        return dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            masks=gt_masks,
            bbox_iscrowd=gt_bboxex_iscrowd
        )

    def _decodeMasks(self,img_info,ann_info):
        masks = ann_info['masks']
        h,w = img_info['height'],img_info['width']
        new_mask = [self._poly2mask(mask,h,w) for mask in masks]
        return new_mask

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

class Visualize():

    def padding(self,img, size, divsor=32):
        img_h, img_w, _ = img.shape

        h_rate = size[0] * 1.0 / img_h
        w_new = int(img_w * h_rate)
        img = cv2.resize(img, (w_new, size[0]))

        if w_new // divsor != 0:
            new_to_pad = w_new % divsor
            print('new_to_pad', new_to_pad)
            img = np.pad(img, ((new_to_pad // 2, new_to_pad - new_to_pad // 2), (0, 0), (0, 0)))

        return img

    def generate_grids(self, img, grids=[]):
        h,w,c = img.shape
        for g in grids:
            h_per = int(h*1.0 / g)
            w_per = int(w*1.0 / g)

            # print(h,g)
            # print('h_per',h_per)
            # print('w_per',w_per)

            for cnt in range(1,g):
                img[h_per*cnt:h_per*cnt+1,...] = [255,255,255]

            for cnt in range(1,g):
                img[:,w_per * cnt:w_per * cnt + 1, ...] = [255, 255, 255]

        return img.copy()

    def drawMask(self,img,masks):
        for mask in masks:
            alpha = 0.5
            color = (0, 0.6, 0.6)
            threshold = 0.5
            mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)
            for c in range(3):  # 3个通道
                # mask=1执行前一个，否则后一个
                img[:, :, c] = np.where(mask == 1,
                                          img[:, :, c] *
                                          (1 - alpha) + alpha * color[c] * 255,
                                          img[:, :, c])
        return img


    # 可视化图像
    def draw(self,img_meta,with_box=True,with_mask=False,with_grids=False):
        print(img_meta)
        img_path = img_meta['img_info']['file_name']
        img = cv2.imread(img_path)
        # img = self.padding(img,(768,512),32)
        if with_box:
            bboxs_info = img_meta['ann_info']['bboxes']
            for n_box in range(bboxs_info.shape[0]):
                x1,y1,x2,y2 = bboxs_info[n_box]
                cv2.rectangle(img,(x1,y1),(x2,y2),(0, 255, 0), 2)
        if with_grids:
            img = self.generate_grids(img,[12]) #[40,36,24,16,12]

        if with_mask:
            img = self.drawMask(img,img_meta['ann_info']['masks'])
        return img

if __name__ == '__main__':
    dataset = readConfig('config/coco.yaml')
    coco = COCODataset(**dataset['val_data'], training=False)
    vis = Visualize()
    while True:
        img = vis.draw(coco.get_one_img(),with_box=True,with_grids=True,with_mask=False)
        cv2.imshow('src', img)
        chr = cv2.waitKey(0)&0xff
        if chr==ord('q'):
            break
