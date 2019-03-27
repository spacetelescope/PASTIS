"""
Directly copied from hicat code.
"""

from __future__ import (absolute_import, division,
                        unicode_literals)

# noinspection PyUnresolvedReferences
from builtins import *

import configparser
import os


config_file_name = "config.ini"
override_file_name = "config_local.ini"


def get_config_ini_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), config_file_name)


def load_config_ini():
    # Locate where on disk this file is.
    code_directory = os.path.dirname(os.path.realpath(__file__))

    # Read config file once here.
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()

    # Check if there is a local override config file (which is ignored by git).
    local_override_path = os.path.join(code_directory, override_file_name)
    result = config.read(local_override_path)
    if not result:
        config.read(os.path.join(code_directory, config_file_name))
    return config


# Import CONFIG_INI below instead of loading the ini file manually.
CONFIG_INI = load_config_ini()
