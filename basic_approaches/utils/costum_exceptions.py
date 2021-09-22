import logging


class FailedAugmentation(Exception):
    def __init__(self, msg=""):
        self.msg = msg
        logging.debug(msg)  # use your logging things here

    def __str__(self):
        return self.msg


class ShutdownException(Exception):
    pass
