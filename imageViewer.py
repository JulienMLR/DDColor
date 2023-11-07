import cv2
import tkinter
import time
import threading
from PIL import Image, ImageTk

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
        # self.tkroot = tkinter.Tk()
        # w, h = self.tkroot.winfo_screenwidth(), self.tkroot.winfo_screenheight()
        # self.tkroot.overrideredirect(1)
        # self.tkroot.geometry("%dx%d+0+0" % (w, h))
        # self.tkroot.focus_set()
        # canvas = tkinter.Canvas(self.tkroot, width=w,height=h)
        # canvas.pack()
        # canvas.configure(background='black')
        while self.run:
            if self.image is not None:
                # pilImage = Image.fromarray(self.image, "LAB")
                # image = ImageTk.PhotoImage(pilImage)
                # imagesprite = canvas.create_image(w/2,h/2,image=image)
                # self.tkroot.update_idletasks()
                # self.tkroot.update()

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