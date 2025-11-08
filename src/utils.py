import os
import logging



class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    On_IBlue = '\033[0;104m'
    On_Cyan = '\033[46m'
    BICyan = '\033[1;96m'
    reset = "\x1b[0m"
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    FORMATS = {
        logging.DEBUG: yellow + format + reset,
        logging.INFO: BICyan + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger_instances = {}
app_name = os.environ.get('APP_NAME', 'ML_App')


def get_logger_by_name(module_name, file_name=None, log_level="DEBUG"):
    global logger_instances
    if not file_name:
        # file_name = os.path.join(os.path.dirname(
        #     __file__), '/shared_data/logs/{}.log'.format(app_name))
        file_name = os.path.join(os.path.dirname(
            __file__), '../logs/{}.log'.format(app_name))
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
    logging_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if module_name not in logger_instances:
        logger_instances[module_name] = logging.getLogger(module_name)
        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(CustomFormatter())
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(CustomFormatter())
        logger_instances[module_name].addHandler(stream_handler)
        logger_instances[module_name].addHandler(file_handler)
    logger_instances[module_name].setLevel(getattr(logging, log_level))

    return logger_instances[module_name]
