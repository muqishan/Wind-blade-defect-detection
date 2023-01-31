import os
import cv2
from preprocess import *

'''
由于ng中，并不是每一个叶子都是注塑不满的情况，以0表示该叶子ok，1表示该叶子ng，则具有以下情况
0 0 0 0 0 1 0 0 1 0 0
所以，ng中每一个叶子，不能都作为注塑不满的ng标签
取8张ok，4张ng作为数据集
其中4张ng对每个叶子进行手工标注是否为注塑不满
'''
label = {'OK': 0,
         'NG': 1
         }
BASE_DIR = os.getcwd()


class loaddata():
    def __init__(self):
        self.c = 0  # 如果需要新加入数据集，更改c即可，避免文件名冲突

    def get_data(self, path):
        ok_paths = os.listdir(path)
        ok_paths.sort(key=lambda x: int(x[:-4]))
        data, data_label = [], []
        for ok_path in ok_paths:
            ok_path = os.path.join('OK', ok_path)
            ok_img = cv2.imread(ok_path, cv2.IMREAD_GRAYSCALE)
            # 获取ROI
            img = scharr(ok_img)
            y = getROI(img)
            for idx, i in enumerate(y):
                if idx == 0:
                    continue
                else:
                    roi, s_img, _ = i
                    r = cv2.resize(roi, (256, 256))
                    data.append(r)
                    data_label.append(label['OK'])
            data = data[:-1]
            data_label = data_label[:-1]
        return data, data_label

    def make_ng(self):
        """
        此处NG仅作为训练使用,因此获取的roi直接标注出NG的叶子，
        生成图片后需要手工标注出ng
        标注方式为，在对应注塑不满叶子的图片名中，添加ng
        如1.png为注塑不满的叶子
        将1.png重命名为1ng.png
        并移动到ng_train文件夹下
        """
        ok_paths = os.listdir('NG')   # 整个NG文件夹视为训练集
        ok_paths.sort(key=lambda x: int(x[:-4]))
        for ng_path in ok_paths:
            print(ng_path)
            ng_path = os.path.join('NG', ng_path)
            ng_img = cv2.imread(ng_path, cv2.IMREAD_GRAYSCALE)
            img = scharr(ng_img)
            y = getROI(img)
            '''
            需要移除每个图片最上方及最下方的叶子
            '''
            for idx, i in enumerate(y):
                if idx == 0:
                    continue
                else:
                    roi, s_img, _ = i
                    self.c += 1
                    cv2.imwrite(str(self.c) + '.png', roi)

            os.remove(str(self.c) + '.png')
            self.c -= 1

    def get_ng(self):
        '''
        获取ng_train文件夹下所有标注好的训练集
        检测文件名中是否具有‘ng’来判别图片类别
        '''
        data, data_label = [], []
        ng_paths = os.listdir('ng_train')
        # ng_paths.sort(key=lambda x: int(x[:-4]))
        for path_ in ng_paths:
            path = os.path.join('ng_train', path_)
            roi = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            r = cv2.resize(roi, (256, 256))
            data.append(r)
            if 'ng' in path_[:-4]:
                data_label.append(label['NG'])
            else:
                data_label.append(label['OK'])
        return data, data_label


if __name__ == '__main__':

    l = loaddata()
    # l.get_data('OK')
    l.get_ng()
