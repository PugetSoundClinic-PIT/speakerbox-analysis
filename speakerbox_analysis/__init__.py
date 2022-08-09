# -*- coding: utf-8 -*-

"""Top-level package for speakerbox_analysis."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("speakerbox-analysis")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Eva Maxfield Brown"
__email__ = "evamaxfieldbrown@gmail.com"
