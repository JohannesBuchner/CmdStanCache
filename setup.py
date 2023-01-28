#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools
import os
import io

here = os.path.abspath(os.path.dirname(__file__))

DESCRIPTION = "Smart cache for Stan models and runs"
try:
    with io.open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

stan_cache = os.path.expanduser("~/.stan_cache")

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        print("Creating %s" % file_path)
        os.makedirs(file_path)

setuptools.setup(
    name="cmdstancache",
    version="1.2.1",
    description="Smart cache for Stan models and runs",
    long_description=long_description,
    author="Johannes Buchner",
    author_email="johannes.buchner.acad@gmx.com",
    python_requires='>3.0, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    url="https://github.com/JohannesBuchner/CmdStanCache",
    py_modules=['cmdstancache'],
    install_requires=["cmdstanpy", "joblib"],
    extras_require=dict(plot=['matplotlib', 'corner']),
    setup_requires=["pytest-runner", ],
    test_suite='tests',
    tests_require=["pytest>=3", "matplotlib", "corner"],
    include_package_data=True,
    license="GPL",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)


ensure_dir(stan_cache)
