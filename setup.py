#!/usr/bin/env python3
import setuptools

VERSION = '0.9.0'

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='validating_models',
    version=VERSION,
    description='Supporting Explainable AI on Semantic Constraint Validation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GNU/GPLv3',
    author='Julian Gercke',
    url='https://github.com/JulianLoewe/Validating_Models',
    download_url='https://github.com/JulianLoewe/Validating_Models/archive/refs/tags/v' + VERSION + '.tar.gz',
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
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research'
    ]
)
