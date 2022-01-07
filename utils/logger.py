import numpy as np
import logging
import random

def init_log(logname=__name__, filename = "log_file.log", level=logging.DEBUG, console=True):
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=level,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M',
                        filename=filename,
                        filemode='w')
    # Now, define a couple of other loggers which might represent areas in your
    # application:
    logger = logging.getLogger(logname)
    if console:
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the the current logger
        logger.addHandler(console)
    return logger

