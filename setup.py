import sys

from setuptools import setup

if sys.version_info.major != 3:
    print("This module is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

setup(
    name='ui_adapt',
    author='Dagasfi',
    version='0.0.1',
    install_requires=[
        'gym==0.10.5',
        'numpy==1.13.3',
        'matplotlib',
    ]
)