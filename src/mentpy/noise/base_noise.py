from abc import ABCMeta, abstractmethod


class BaseNoise(metaclass=ABCMeta):
    """BaseNoise clase"""

    def __init__(self):
        """Initialize the current object"""
