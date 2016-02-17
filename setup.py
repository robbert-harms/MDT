#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import re
import platform
from setuptools import setup, find_packages


def load_requirements(fname):
    is_comment = re.compile('^\s*(#|--).*').match
    with open(fname) as fo:
        return [line.strip() for line in fo if not is_comment(line) and line.strip()]

with open('README.rst', 'rt') as f: readme = f.read()
with open('HISTORY.rst', 'rt') as f: history = f.read().replace('.. :changelog:', '')
with open('mdt/__init__.py') as f: version_file_contents = f.read()

requirements = load_requirements('requirements.txt')
requirements_tests = load_requirements('requirements_tests.txt')

ver_dic = {}
exec(compile(version_file_contents, "mdt/__init__.py", 'exec'), ver_dic)

# scripts = glob.glob('bin/mdt-*')
# if platform.system() == 'Windows':
#     scripts = list(scripts) + list(glob.glob('bin_windows_extra/mdt-*.bat'))

setup(
    name='mdt',
    version=ver_dic["VERSION"],
    description='A diffusion toolkit for parallelized sampling and optimization of diffusion data.',
    long_description=readme + '\n\n' + history,
    author='Robbert Harms',
    author_email='robbert.harms@maastrichtuniversity.nl',
    url='https://github.com/robbert-harms/mdt',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license="LGPL v3",
    zip_safe=False,
    keywords='mdt',
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering'
    ],
    test_suite='tests',
    tests_require=requirements_tests,
    entry_points={
        'console_scripts': [
            'mdt-list-devices = mdt.cl_scripts.mdt_list_devices:ListDevices.console_script',
        ],
        'gui_scripts': [
            'mdt-gui-single = mdt.cl_scripts.mdt_gui_single:GUISingle.console_script',
        ]
    }
)
