import codecs
from os import path
from setuptools import find_packages
from setuptools import setup


here = path.abspath(path.dirname(__file__))

with codecs.open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = ['gin-config >= 0.1.1', 'absl-py >= 0.2.2',
                    'tensorflow==1.15rc3', 'opencv-python >= 3.4.1.15',
                    'gym >= 0.10.5', 'dopamine-rl==1.0.2']

log_dqn_description = (
    'LogDQN agent from van Seijen, Fatemi, Tavakoli (2019)')

setup(
    name='log_dqn',
    version='0.0.1',
    packages=find_packages(),
    author_email='a.tavakoli@imperial.ac.uk',
    install_requires=install_requires,
    description=log_dqn_description,
    long_description=long_description,
    long_description_content_type='text/markdown'
)