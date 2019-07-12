# Configure application logger
import logging

ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(name)-30s | %(levelname)-5s : %(message)s")

ch.setFormatter(formatter)
logger = logging.getLogger("main")
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)
