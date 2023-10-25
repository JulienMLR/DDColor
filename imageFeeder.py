import socket
import numpy
import time
import threading

class ImageListener:
    def onImage(self, image):
        pass

class ImageFeeder:

    def __init__(self, ip, port):
        self.listeners = []
        self.listenersMutex = threading.Lock()
        self.broadcaster_ip = ip
        self.broadcaster_port = port
        self.receptionThread = None
        self.run = False

    def addListener(self, listener):
        self.listenersMutex.acquire()
        self.listeners.append(listener)
        self.listenersMutex.release()

    def removeListener(self, listener):
        self.listenersMutex.acquire()
        self.listeners.remove(listener)
        self.listenersMutex.release()

    def receptionLoop(self):
        while self.run:
            try:
                clientSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
                clientSock.connect((self.broadcaster_ip, self.broadcaster_port))
                while self.run: 
                    height = int.from_bytes(clientSock.recv(4, socket.MSG_WAITALL), 'big', signed=False)
                    width = int.from_bytes(clientSock.recv(4, socket.MSG_WAITALL), 'big', signed=False)
                    if height <=0 or width <= 0:
                        raise ConnectionRefusedError
                    size = height*width
                    pixels = bytearray(clientSock.recv(size, socket.MSG_WAITALL))
                    if len(pixels) < size:
                        raise ConnectionRefusedError
                    data = numpy.array(pixels).reshape((height, width))
                    image = data
                    self.listenersMutex.acquire()
                    for listener in self.listeners:
                        listener.onImage(image)
                    self.listenersMutex.release()
                clientSock.shutdown(socket.SHUT_RDWR)
                clientSock.close()
            except ConnectionRefusedError:
                time.sleep(1)

    def start(self):
        self.run = True
        self.receptionThread = threading.Thread(target= self.receptionLoop, args=())
        self.receptionThread.start()

    def stop(self):
        self.run = False
        self.receptionThread.join()
