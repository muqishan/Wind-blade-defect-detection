import cv2
import numpy as np
import os

BASE_DIR = os.getcwd()

test_ok = r'OK\1124164721624.bmp'

test_ng = r'NG\1205171545118.bmp'

ok_img = cv2.imread(test_ok, cv2.IMREAD_GRAYSCALE)

ng_img = cv2.imread(test_ng, cv2.IMREAD_GRAYSCALE)

kernel = np.ones((7, 7), np.uint8)

def scharr(img):
    # 2 scharr
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=-1)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=-1)
    # 3 将数据进行转换
    Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
    Scale_absY = cv2.convertScaleAbs(y)
    # 4 结果合成
    result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
    _, thresh = cv2.threshold(result, 75, 255, cv2.THRESH_BINARY)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # 闭操作

    '''
    保存某个过程
    debug可用，松开得到   ------------------------------------------------------------
    '''
    # cv2.imwrite('lll.bmp', closing)

    return closing

def getROI(img: np.ndarray) -> np.ndarray:
    """
    cv2.RETR_EXTERNAL：输出轮廓中只有外侧轮廓信息；
    cv2.RETR_LIST：以列表形式输出轮廓信息，各轮廓之间无等级关系；
    cv2.RETR_CCOMP：输出两层轮廓信息，即内外两个边界（下面将会说到contours的数据结构）；
    cv2.RETR_TREE：以树形结构输出轮廓信息。

    cv2.CHAIN_APPROX_NONE：存储轮廓所有点的信息，相邻两个轮廓点在图象上也是相邻的；
    cv2.CHAIN_APPROX_SIMPLE：压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标；
    cv2.CHAIN_APPROX_TC89_L1：使用teh-Chinl chain 近似算法保存轮廓信息。

    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    for idx, cnt in enumerate(contours):
        # 绘制轮廓
        x, y, w, h = cv2.boundingRect(cnt)
        roi_image = img[y:y + h, x:x + w]  # 裁剪 roi
        # cv2.imwrite('1.png', roi_image)

        # 当闭操作的卷积核过大时，会出现预期外的融合，常识上宽度低于1000的ROI是非叶子轮廓，因此丢弃
        if roi_image.shape[1] < 1000:
            continue
        else:
            yield roi_image, img,(x, y, w, h)
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # cv2.imwrite(os.path.join('process', str(idx)+'.bmp'), img) # 随便保存几张看一下效果
    # cv2.imshow("Result", ok_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return

if __name__ == '__main__':

    binary = scharr(ok_img)
    result = getROI(binary)
    next(result)
