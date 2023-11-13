import numpy as np
import cv2
import time


def test():
    RGB = [0.09754902, 0.09754902, 0.09754902]
    XYZ = [0, 0, 0, ]

    X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
    Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
    Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
    XYZ[0] = round(X, 4)
    XYZ[1] = round(Y, 4)
    XYZ[2] = round(Z, 4)

    # Observer= 2°, Illuminant= D65
    XYZ[0] = float(XYZ[0]) / 95.047         # ref_X =  95.047
    XYZ[1] = float(XYZ[1]) / 100.0          # ref_Y = 100.000
    XYZ[2] = float(XYZ[2]) / 108.883        # ref_Z = 108.883

    num = 0
    for value in XYZ:

        if value > 0.008856:
            value = value ** (0.3333333333333333)
        else:
            value = (7.787 * value) + (16 / 116)

        XYZ[num] = value
        num = num + 1

    Lab = [0, 0, 0]

    L = (116 * XYZ[1]) - 16
    a = 500 * (XYZ[0] - XYZ[1])
    b = 200 * (XYZ[1] - XYZ[2])

    Lab[0] = round(L, 4)
    Lab[1] = round(a, 4)
    Lab[2] = round(b, 4)

    print(Lab)


def TEST_runIR2RGB(image):
        test1Img = np.copy(image)
        test2Img = np.copy(image)
        # HAS DONE BY MODEL**
        start = time.time()
        test1Img = cv2.cvtColor(test1Img,cv2.COLOR_GRAY2BGR)
        test1Img = (test1Img / 255.0).astype(np.float32)
        #orig_l = cv2.cvtColor(test1Img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

        # resize rgb image -> lab -> get grey -> rgb
        test1Img = cv2.resize(test1Img, (512, 512))
        print(test1Img[0][0])
        test1Img_l = cv2.cvtColor(test1Img, cv2.COLOR_BGR2Lab)[:, :, :1]
        print(test1Img_l)
        #test1Img_gray_lab = np.concatenate((test1Img_l, np.zeros_like(test1Img_l), np.zeros_like(test1Img_l)), axis=-1)
        #test1Img_gray_rgb = cv2.cvtColor(test1Img_gray_lab, cv2.COLOR_LAB2RGB)
        stop = time.time()
        processingtime = stop - start
        print("Method 1 timing", processingtime)
        # HAS WE WANT TO OPTIMIZE IT
        start = time.time()
        test2Img = (test2Img / 255.0).astype(np.float32)
        test2Img = cv2.resize(test2Img, (512, 512))
        print(test2Img[0][0])
        test2Img = (test2Img/100.0)
        f = lambda y:  116*(y**(1/3))-16 if y > 0.008856 else 903.3*y
        f_arr = np.vectorize(f)
        test2Img_l = f_arr(test2Img).astype(np.float32)
        print(test2Img_l)
        #test2Img_gray_lab = np.concatenate((test2Img_l, np.zeros_like(test2Img_l), np.zeros_like(test2Img_l)), axis=-1)
        #test2Img_gray_rgb = cv2.cvtColor(test2Img_gray_lab, cv2.COLOR_LAB2RGB)
        stop = time.time()
        processingtime = stop - start
        print("Method 2 timing", processingtime)
        return image