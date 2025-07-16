# src/fpose/setup.py
from setuptools import setup, find_packages

setup(
    name='models',
    version='0.1',
    package_dir={'models': 'models'},                  
    packages=find_packages(where='models', include=['models*']),
)