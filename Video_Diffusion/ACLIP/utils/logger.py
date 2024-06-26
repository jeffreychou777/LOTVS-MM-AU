import os
import sys
import logging
import functools
from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    # if dist_rank == 0:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, 'log_rank.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def get_logger(filename):
    """
    获取logger对象
    :param filename:log文件路径
    :return:
    """
    path, _ = os.path.split(filename)
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel('INFO')
    basic_format = "%(asctime)s:%(levelname)s:%(message)s"
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(basic_format, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel('INFO')
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
