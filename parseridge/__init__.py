# Configure application logger
import logging
import subprocess

ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s : %(message)s")

ch.setFormatter(formatter)
logger = logging.getLogger("main")
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

try:
    process = subprocess.Popen(
        ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
    )
    git_commit = process.communicate()[0].strip().decode()
except ValueError or IndexError:
    git_commit = None
