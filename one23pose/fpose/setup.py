from setuptools import setup, find_packages

setup(
    name='fpose',
    version='0.1',
    package_dir={'fpose': 'fpose'},                  
    packages=find_packages(where='fpose', include=['fpose*']),
)