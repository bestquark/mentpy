# Copyright (C) [2023] Luis Mantilla
#
# This program is released under the GNU GPL v3.0 or later.
# See <https://www.gnu.org/licenses/> for details.
from abc import ABCMeta, abstractmethod


class BaseNoise(metaclass=ABCMeta):
    """BaseNoise clase"""

    def __init__(self):
        """Initialize the current object"""
