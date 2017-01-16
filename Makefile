.PHONY: clean clean-build clean-pyc clean-test lint test tests test-all coverage docs docs-pdf docs-man release dist install uninstall dist-ubuntu _package-ubuntu

PYTHON=$$(which python3)
PIP=$$(which pip3)
PROJECT_NAME=mdt
PROJECT_VERSION=$$($(PYTHON) setup.py --version)
GPG_SIGN_KEY=0E1AA560
UBUNTU_MAIN_TARGET_DISTRIBUTIONS=xenial
UBUNTU_OTHER_TARGET_DISTRIBUTIONS=yakkety


help:
	@echo "clean - remove all build, test, coverage and Python artifacts (no uninstall)"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "tests - synonym for test"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "docs-pdf - generate the PDF documentation, including API docs"
	@echo "docs-man - generate the linux manpages"
	@echo "release - package and upload a release"
	@echo "dist - create a pip package"
	@echo "dist-ubuntu - create an Ubuntu package"
	@echo "install - installs the package using pip"
	@echo "uninstall - uninstalls the package using pip"

clean: clean-build clean-pyc clean-test
	$(PYTHON) setup.py clean

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

lint:
	flake8 $(PROJECT_NAME) tests

test:
	$(PYTHON) setup.py test

tests: test

test-all:
	tox

coverage:
	coverage run --source $(PROJECT_NAME) setup.py test
	coverage report -m
	coverage html
	@echo "To view results type: htmlcov/index.html &"

docs:
	rm -f docs/$(PROJECT_NAME)*.rst
	rm -f docs/modules.rst
	$(MAKE) -C docs clean
	sphinx-apidoc -o docs/ $(PROJECT_NAME)
	$(MAKE) -C docs html SPHINXBUILD='python3 $(shell which sphinx-build)'
	@echo "To view results type: firefox docs/_build/html/index.html &"

docs-pdf:
	rm -f docs/$(PROJECT_NAME)*.rst
	rm -f docs/modules.rst
	$(MAKE) -C docs clean
	sphinx-apidoc -o docs/ $(PROJECT_NAME)
	$(MAKE) -C docs latexpdf SPHINXBUILD='python3 $(shell which sphinx-build)'
	@echo "To view results use something like: evince docs/_build/latex/mdt.pdf &"

docs-man:
	rm -f docs/$(PROJECT_NAME)*.rst
	rm -f docs/modules.rst
	$(MAKE) -C docs clean
	sphinx-apidoc -o docs/ $(PROJECT_NAME)
	$(MAKE) -C docs man SPHINXBUILD='python3 $(shell which sphinx-build)'
	@echo "To view results use something like: man docs/_build/man/mdt.1 &"


# todo: add GitHub Releases API hook here
release: clean release-ubuntu-ppa release-pip

release-pip:
	$(PYTHON) setup.py sdist upload
	$(PYTHON) setup.py bdist_wheel upload

release-ubuntu-ppa: dist-ubuntu
	dput ppa:robbert-harms/cbclab dist/$(PROJECT_NAME)_$(PROJECT_VERSION)-1_source.changes
	for ubuntu_version in $(UBUNTU_OTHER_TARGET_DISTRIBUTIONS) ; do \
		dput ppa:robbert-harms/cbclab dist/$(PROJECT_NAME)_$(PROJECT_VERSION)-1~$${ubuntu_version}1_source.changes ; \
	done

dist: clean
	$(PYTHON) setup.py sdist
	$(PYTHON) setup.py bdist_wheel
	ls -l dist

dist-ubuntu: clean
	$(PYTHON) setup.py sdist
	cp dist/$(PROJECT_NAME)-$(PROJECT_VERSION).tar.gz dist/$(PROJECT_NAME)_$(PROJECT_VERSION).orig.tar.gz
	tar -xzf dist/$(PROJECT_NAME)-$(PROJECT_VERSION).tar.gz -C dist/
	$(MAKE) _package-ubuntu suite=$(UBUNTU_MAIN_TARGET_DISTRIBUTIONS) debian-version=1 build-flag=-sa

	for ubuntu_version in $(UBUNTU_OTHER_TARGET_DISTRIBUTIONS) ; do \
		$(MAKE) _package-ubuntu suite=$$ubuntu_version debian-version=1~$${ubuntu_version}1 build-flag=-sd ; \
	done

_package-ubuntu:
	# Requires the following parameters to be set:
	# suite: the suite argument in the debianize command
	# debian-version: the debian-version argument in the debianize command
	# build-flag: the build flag for the debuild command. Commonly something like '-sa' or '-sd'

	rm -rf debian/source
	$(PYTHON) setup.py --command-packages=stdeb.command debianize --suite $(suite) --debian-version $(debian-version) --with-python2 False --with-python3 True
	$(PYTHON) setup.py prepare_debian_dist
	rm -rf dist/$(PROJECT_NAME)-$(PROJECT_VERSION)/debian/
	cp -r debian dist/$(PROJECT_NAME)-$(PROJECT_VERSION)/
	cd dist/$(PROJECT_NAME)-$(PROJECT_VERSION)/; dpkg-source -b .
	cd dist/$(PROJECT_NAME)-$(PROJECT_VERSION)/; debuild -S $(build-flag) -k$(GPG_SIGN_KEY)

install: dist
	$(PIP) install --upgrade --no-deps --force-reinstall dist/$(PROJECT_NAME)-*.tar.gz

uninstall:
	$(PIP) uninstall -y $(PROJECT_NAME)
