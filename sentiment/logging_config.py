import os
from os.path import *
import __main__
import logging
from logging.config import dictConfig


# At the top-most level of module, in __init__.py, import and run init_configs(): 
    # from *logging_config import init_configs
    # init_configs()
# After which, it is possible from any script in this module, to do:
    # import logging
    # logger = logging.getLogger(__name__)
# For each submodule level (e.g. autoit, pdf, ...), it is possible to define logger in logging_config dictionary below with custom handlers/formatting, 
# just define name of logger with hierarchy of module levels in dot notation (e.g. sentiment.pdf, sentiment.<sub folder name>, etc...),
# and when any script from within the sub module does 
    # logger = logging.getLogger('__name__')
# , it will automatically get the correct logger version as defined. 


##############################################################################################################

log_name = 'sentiment.log'

# Defining log_path:
# Default behaviour is to put log beside executing ("calling") script that imports sentiment.
# If calling script is ipykernel, log file is placed in package folder, and user is notified of location. 
if '__file__' in dir(__main__):
    _calling_script = __main__.__file__
    _calling_script_dir = dirname(_calling_script)
    _calling_script_filename = basename(_calling_script)

    # puts log file beside execution script
    log_path = join(_calling_script_dir, log_name)
else:
    # this happens with ipykernel, there is no "calling script", so just put a placeholder for logging
    _calling_script_filename = '-' * 10

    # puts log file in sentiment/logs/
    log_path = join(dirname(__file__), f'logs/{log_name}')

    # make log directory if not exists
    log_dir = dirname(log_path)
    if not isdir(log_dir):
        os.mkdir(log_dir)

    # notify user of log location
    print(f'NOTICE: sentiment log file will be at {log_path}')




logging_config = {
    'version': 1,
    'formatters': {
        'default': {
            'format': f'[{_calling_script_filename} | ' + '%(name)s] : %(asctime)s : %(levelname)s : %(message)s'
        }
    },
    'handlers': {
        'pdf_file_handler': {
            "level": "INFO",
            "formatter": "default",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": log_path,
            "mode": "a",
            "maxBytes": 10*1024*1024,
            "backupCount": 2
        }
    },
    'loggers': {
        'sentiment.pdf': {
            'handlers': ['pdf_file_handler'],
            'level': 'INFO',
            'propagate': False
        }
    }
}


def init_configs():
        
    dictConfig(logging_config)
