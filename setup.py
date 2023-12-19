import os
import subprocess
import sys

from setuptools import setup, find_packages


setup(
    name="YouRong",
    version="0.0.1",
    packages=find_packages(include=['oft', 'oft.*']),
)
