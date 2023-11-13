import threading
import time
import queue


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

        self.nbFrame = 0
        self.startMesure = None
        th = threading.Thread(target=self.computeFPS)
        th.setDaemon(True)
        th.start()

        # Thread for getting the image + Queue
        self.queue = queue.Queue(maxsize=50)
        th_image_reception = threading.Thread(target=self.process_image)
        th_image_reception.setDaemon(True)
        th_image_reception.start()
        print(vars(self))


    def onImage(self, image):
        self.queue.put(image)

    def process_image(self):
        while True:
            image = self.queue.get()
            processedImage = self.runIR2RGB(image)
            self.nbFrame += 1
            self.listenersMutex.acquire()
            for listener in self.listeners:
                listener.onImage(processedImage)
            self.listenersMutex.release()
            self.queue.task_done()

    
    def computeFPS(self):
        while True:
            initFrame = self.nbFrame
            time.sleep(1)
            finalFrame = self.nbFrame
            print("FPS ? ", finalFrame-initFrame)


    def addListener(self, listener):
        self.listenersMutex.acquire()
        self.listeners.append(listener)
        self.listenersMutex.release()


    def removeListener(self, listener):
        self.listenersMutex.acquire()
        self.listeners.remove(listener)
        self.listenersMutex.release()


    # def onImage(self, image):
    #     processedImage = self.runIR2RGB(image)
    #     self.nbFrame += 1
    #     self.listenersMutex.acquire()
    #     for listener in self.listeners:
    #         listener.onImage(processedImage)
    #     self.listenersMutex.release()


    def runIR2RGB(self, image):
        if self.startMesure is None:
            self.startMesure = time.time()
        processedimage = self.colorizer.process(img=image)
        return processedimage
    
