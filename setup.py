from setuptools import setup, find_packages
import os

# 读取README.md作为长描述
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='train_dit',
    version='0.1.0',
    author='Yiming Shi',
    author_email='yimingshi666@gmail.com',
    description='A package to occupy GPU resources by performing matrix multiplications.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SKDDJ/train_dit', 
    packages=find_packages(),
    install_requires=[
        'torch>=1.0.0',
    ],
    entry_points={
        'console_scripts': [
            'train_dit=train_dit.core:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
