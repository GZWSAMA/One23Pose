from setuptools import find_packages, setup

setup(
    name='sam2',
    version='0.1.0',
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    packages=find_packages(exclude="notebooks")
)