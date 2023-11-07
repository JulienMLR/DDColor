from imageFeeder import ImageListener
import queue
import threading

class ImageGetter(ImageListener):
    def __init__(self) -> None:
        pass


# class picture_getter(threading.Thread):
#     def __init__(self, url, picture_queue):
#         self.url = url
#         self.picture_queue = picture_queue
#         super(picture_getter, self).__init__()

#     def run(self):
#         print("Starting download on " + str(self.url))
#         self._get_picture()

#     def _get_picture(self):
#         # --- get your picture --- #
#         self.picture_queue.put(picture)

