# src/fpose/setup.py
from setuptools import setup, find_packages

setup(
    name='trellis',
    version='0.1',
    package_dir={'trellis': 'trellis'},                  
    packages=find_packages(where='trellis', include=['trellis*']),
)