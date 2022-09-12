import importlib.metadata
import logging

import coloredlogs

from .cli import diffuse, diffuse_cli  # noqa: F401

logging.basicConfig(format="%(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")
__version__ = importlib.metadata.version("stable_diffusion_cli")
