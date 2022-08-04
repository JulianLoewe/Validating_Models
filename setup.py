#!/usr/bin/env python3
import setuptools

setuptools.setup(
    name='validating_models',
    version='x.y.z',
    description='validating_models',
    author='Julian Gercke',
    packages=setuptools.find_packages(),
    install_requires=[
        'shaclapi>=0.9.0',
        'cairosvg>=2.5.2',
        'dtreeviz',
        'matplotlib',
        'numpy',
        'openml',
        'pandas',
        'Pillow',
        'rdflib>=6.1.0',
        'requests',
        'scikit_learn',
        'seaborn>=0.11.2',
        'SPARQLWrapper>=2.0.0',
        'tqdm>=4.63.0',
        'pebble>=4.6.0'
    ],
    include_package_data=False,
    python_requires='>=3.8'
)
