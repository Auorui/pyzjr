import os
import logging
from pyzjr.data.utils import timestr

def logdir(dir="logs", simple=True, prefix="", suffix=""):
    """
    Logging generator
    :param dir: Default "logs"
    :param simple: '%Y_%m_%d_%H_%M_%S' or '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    :param prefix: Add folder name before time string
    :param suffix: After adding the folder name to the time string
    :return:
    """
    time_str = timestr(simple)
    folder_names = [prefix, time_str, suffix]
    folder_names = [folder for folder in folder_names if folder]
    log_dir = os.path.join(dir, *folder_names)
    os.makedirs(log_dir, exist_ok=True)

    return log_dir


def get_logger(filepath='./exp.log', verbosity=1, name=None):
    """
    获取一个日志记录器

    Args:
        filepath (str, optional): 日志文件的路径，默认为 './exp.log'。
        verbosity (int, optional): 日志输出的详细程度，0 表示调试级别，1 表示信息级别，2 表示警告级别，默认为 1。
        name (str, optional): 日志记录器的名称，默认为 None。

    Returns:
        logging.Logger: 日志记录器对象。
    """
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s]%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filepath, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger




if __name__=="__main__":
    logger = get_logger()
    logger.debug('This is a DEBUG message.')
    logger.info('This is an INFO message.')
    logger.warning('This is a WARNING message.')
    try:
        result = 10 / 0
    except Exception as e:
        logger.error(f'An error occurred: {str(e)}', exc_info=True)
    logger.critical('This is a CRITICAL message.')

