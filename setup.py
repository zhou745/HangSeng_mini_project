from setuptools import setup, find_packages

setup(
    name='csi_proj',
    version='1.0.0',
    description='csi_proj proj packages',
    packages=['utils', 'model','modules','dataset'],
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)