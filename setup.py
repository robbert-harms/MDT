#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import glob
from textwrap import dedent
import os
import sys
from setuptools import setup, find_packages, Command


def load_requirements(fname):
    is_comment = re.compile('^\s*(#|--).*').match
    with open(fname) as fo:
        return [line.strip() for line in fo if not is_comment(line) and line.strip()]

with open('README.rst', 'rt') as f:
    readme = f.read()

with open('mdt/__version__.py') as f:
    version_file_contents = "\n".join(f.readlines())
    ver_dic = {}
    exec(compile(version_file_contents, "mdt/__version__.py", 'exec'), ver_dic)

requirements = load_requirements('requirements.txt')
requirements_tests = load_requirements('requirements_tests.txt')

long_description = readme
if sys.argv and len(sys.argv) > 3 and sys.argv[2] == 'debianize':
    long_description = dedent("""
        The Maastricht Diffusion Toolbox is a parallized neuro-imaging model recovery toolbox.
        Being GPU accelerated, it allows for high-performance computing in MRI imaging analysis.
    """).lstrip()

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

info_dict = dict(
    name='mdt',
    version=ver_dic["VERSION"],
    description='Maastricht Diffusion Toolbox',
    long_description=long_description,
    author='Robbert Harms',
    author_email='robbert.harms@maastrichtuniversity.nl',
    maintainer='Robbert Harms',
    maintainer_email='robbert.harms@maastrichtuniversity.nl',
    url='https://github.com/cbclab/MDT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license="LGPL v3",
    zip_safe=False,
    keywords='mdt, diffusion MRI, model recovery, imaging analysis',
    classifiers=[
        'Environment :: Console',
        'Environment :: X11 Applications :: Qt',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Development Status :: 5 - Production/Stable',
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


class PrepareDebianDist(Command):
    description = "Prepares the debian dist prior to packaging."
    user_options = []

    def initialize_options(self):
        self.cwd = None

    def finalize_options(self):
        self.cwd = os.getcwd()

    def run(self):
        with open('./debian/rules', 'a') as f:
            f.write('\noverride_dh_auto_test:\n\techo "Skip dh_auto_test"')

        self._set_copyright_file()

    def _set_copyright_file(self):
        with open('./debian/copyright', 'r') as file:
            copyright_info = file.read()

        copyright_info = copyright_info.replace('{{source}}', info_dict['url'])
        copyright_info = copyright_info.replace('{{years}}', '2016-2017')
        copyright_info = copyright_info.replace('{{author}}', info_dict['author'])
        copyright_info = copyright_info.replace('{{email}}', info_dict['author_email'])

        with open('./debian/copyright', 'w') as file:
            file.write(copyright_info)


info_dict.update(cmdclass={'prepare_debian_dist': PrepareDebianDist})
setup(**info_dict)
