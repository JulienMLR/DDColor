import sys
import os
import argparse
import time
import signal

from imageFeeder import ImageFeeder
from imageViewer import ImageViewer
from imageProcessor import ImageProcessor

run = False

def signal_handler(sig, frame):
    global run
    run = False

def main():
    global run
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--broadcaster-ip", help="The broadcaster ip address", default="127.0.0.1")
    parser.add_argument("--broadcaster-port", help="The broadcaster video port", type=int, default=8887)
    parser.add_argument("--model-path", help="The colorization model path", default="~/Documents/DDColor/pretrain/ddcolor_modelscope.pth")
    args = parser.parse_args()

    feeder = ImageFeeder(args.broadcaster_ip, args.broadcaster_port)
    processor = ImageProcessor(modelpath=os.path.expanduser(args.model_path))
    viewer = ImageViewer()

    run = True
    feeder.addListener(processor)
    processor.addListener(viewer)
    feeder.start()
    viewer.start()
    while run and viewer.run:
        time.sleep(0.5)
    viewer.stop()
    feeder.stop()
    feeder.removeListener(processor)
    processor.removeListener(viewer)


if __name__ == '__main__':
    sys.exit(main())