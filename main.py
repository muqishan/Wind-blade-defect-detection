from loaddatas import loaddata as L
from sklearn.ensemble import RandomForestClassifier
from preprocess import *
import os


class Feature(object):
    def __init__(self):
        loaddata = L()
        date1, label1 = loaddata.get_data('OK')
        date2, label2 = loaddata.get_ng()
        self.X_ = date1 + date2
        self.y = label1 + label2
        self.X = None

    def open(self):
        self.X = [i.flatten() for i in self.X_]


class Randf(Feature):
    def __init__(self):
        super().__init__()
        super().open()
        self.clf = RandomForestClassifier()

    # 训练
    def train(self):
        self.clf.fit(self.X, self.y)

    def predect(self, roi):
        return self.clf.predict(roi)

    def test(self):
        '''
         更改imread中路径进行测试
        更改为遍历test文件夹下的图片，
        '''
        imgs = os.listdir('test')
        for img in imgs:
            # img = os.path.join('test', img)
            img_ = cv2.imread('test/ng.bmp', cv2.IMREAD_GRAYSCALE)

            img = scharr(img_)
            y = getROI(img)
            result = []
            for idx, i in enumerate(y):
                if idx == 0:
                    continue
                else:
                    roi, s_img, rect = i
                    r = cv2.resize(roi, (256, 256))  # resize
                    rr = self.predect([r.flatten()])
                    result.append(int(rr[0]))
                    if int(rr[0]) == 1:  # 绘制NG叶子，保存为png
                        x, y, w, h = rect
                        roi_image = img_[y:y + h, x:x + w]
                        cv2.imwrite(str(idx) + '.png', roi_image)
            print(result[:-1])
            if 1 not in result[:-1]:
                print('OK')
            else:
                print('NG')


r = Randf()
r.train()
r.test()
