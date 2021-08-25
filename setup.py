"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'tensorflow-gpu==2.5.1',
    'tensorflow-datasets',
    'numpy',
    'matplotlib']


setup(
    name='probalistic_vq',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('probalistic_vq')],
    description='Probabilistic Vector Quantization In TF 2.1')