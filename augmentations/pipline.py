# coding:utf-8
# datetime:2020/9/24 2:13 下午
# author: haibo

class Compose(object):
    ''' trans is a list of transform methods '''
    def __init__(self,trans):
        self.trans = trans

    def __call__(self,img,bboxes,masks,labels):
        for t in self.trans:
            img, bboxes, masks, labels = t(img,bboxes,masks,labels)
        return img,bboxes,masks,labels

class Pipline(object):
    def __init__(self,pipline_cfg):
        self.transform = []
        for trans in pipline_cfg.keys():
            if trans == 'ReadImage':
                self.transform.append()

        self.compose = Compose(self.transform)

    def __call__(self, img, boxes,masks,labels):
        return self.compose(img,boxes,masks,labels)

if __name__ == '__main__':
    pass