import cv2
import time
import threading

from imageProcessor import ImageProcessorListener

class ImageViewer(ImageProcessorListener):
    def __init__(self) -> None:
        super().__init__()
        self.windowName = "IR2RGB"
        self.image = None
        self.viewerThread = None
        self.run = False

    def onImage(self, image):
        self.image = image
    
    def viewerLoop(self):
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while self.run:
            if self.image is not None:
                cv2.imshow(self.windowName, self.image)
                if cv2.waitKey(33) & 0xFF == ord('q') or cv2.getWindowProperty(self.windowName, cv2.WND_PROP_VISIBLE) < 1:
                    self.run = False
            else:
                time.sleep(0.1)

    def start(self):
        self.run = True
        self.viewerThread = threading.Thread(target= self.viewerLoop, args=())
        self.viewerThread.start()

    def stop(self):
        self.run = False
        self.viewerThread.join()
        cv2.destroyAllWindows()