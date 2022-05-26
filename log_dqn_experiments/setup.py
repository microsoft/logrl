r'''
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
'''

import codecs
from os import path
from setuptools import find_packages
from setuptools import setup


install_requires = ['gin-config >= 0.1.1', 'absl-py >= 0.2.2',
                    'tensorflow==2.7.2', 'opencv-python >= 3.4.1.15',
                    'gym >= 0.10.5', 'dopamine-rl==1.0.2']

log_dqn_description = (
    'LogDQN agent from van Seijen, Fatemi, Tavakoli (2019)')

setup(
    name='log_dqn',
    version='0.0.1',
    description=log_dqn_description,
    author_email='a.tavakoli@imperial.ac.uk',
    url='https://github.com/microsoft/logrl',
    packages=find_packages(),
    install_requires=install_requires
)