#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import inspect
import os
import re
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


def load_entry_points():
    entry_points = {'console_scripts': []}
    for file in glob.glob('mdt/cli_scripts/*.py'):
        module_name = os.path.splitext(os.path.basename(file))[0]
        command_name = module_name.replace('_', '-')

        def get_command_class_name():
            with open(file) as f:
                match = re.search(r'class (\w*)\(', f.read())
                if match:
                    return match.group(1)
                return None

        command_class_name = get_command_class_name()

        if command_class_name is not None:
            script = '{command_name} = mdt.cli_scripts.{module_name}:{class_name}.console_script'.format(
                command_name=command_name, module_name=module_name, class_name=command_class_name)
            entry_points['console_scripts'].append(script)

    return entry_points


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
    entry_points=load_entry_points()
)
