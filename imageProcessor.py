import threading
import cv2
import time
import numpy as np

from imageFeeder import ImageListener
from inference.colorization_pipline import ImageColorizationPipeline

class ImageProcessorListener:
    def onImage(self, image):
        pass

class ImageProcessor(ImageListener):

    def __init__(self, modelpath) -> None:
        super().__init__()
        self.listeners = []
        self.listenersMutex = threading.Lock()
        print("Using model :", modelpath)
        self.colorizer = ImageColorizationPipeline(modelpath)

    def addListener(self, listener):
        self.listenersMutex.acquire()
        self.listeners.append(listener)
        self.listenersMutex.release()

    def removeListener(self, listener):
        self.listenersMutex.acquire()
        self.listeners.remove(listener)
        self.listenersMutex.release()

    def onImage(self, image):
        processedImage = self.runIR2RGB(image)
        self.listenersMutex.acquire()
        for listener in self.listeners:
            listener.onImage(processedImage)
        self.listenersMutex.release()

    def runIR2RGB(self, image):
        start = time.time()
        #preprocessing
        preprocessedimage = cv2.resize(image, (720, 1280))
        preprocessedimage = cv2.cvtColor(preprocessedimage,cv2.COLOR_GRAY2BGR)
        #processing
        processedimage =  self.colorizer.process(preprocessedimage)
        #postprocessing
        postprocessedimage = cv2.rotate(processedimage, cv2.ROTATE_90_COUNTERCLOCKWISE)
        stop = time.time()
        processingtime = stop - start
        print(processingtime)
        return postprocessedimage
    
    def TEST_runIR2RGB(self, image):
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
