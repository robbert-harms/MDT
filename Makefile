.PHONY: clean clean-build clean-pyc clean-test lint test tests test-all coverage docs release dist install uninstall dist-deb clean-docs

PYTHON=`which python`
PROJECT=mdt

help:
	@echo "clean - remove all build, test, coverage and Python artifacts (no uninstall)"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "clean-docs - remove the binary doc files"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "tests - synonym for test"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "release - package and upload a release"
	@echo "dist - create the package"
	@echo "dist-deb - create a debian package"
	@echo "install - installs the package using pip"
	@echo "uninstall - uninstalls the package using pip"

clean: clean-build clean-pyc clean-test clean-docs
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

clean-docs:
	rm -f docs/$(PROJECT)*.rst
	rm -f docs/modules.rst
	$(MAKE) -C docs clean

lint:
	flake8 $(PROJECT) tests

test:
	$(PYTHON) setup.py test

tests: test

test-all:
	tox

coverage:
	coverage run --source $(PROJECT) setup.py test
	coverage report -m
	coverage html
	@echo "To view results type: htmlcov/index.html &"

docs: clean-docs
	sphinx-apidoc -o docs/ $(PROJECT)
	$(MAKE) -C docs html SPHINXBUILD='python3 $(shell which sphinx-build)'
	@echo "To view results type: firefox docs/_build/html/index.html &"

release: clean
	$(PYTHON) setup.py sdist upload
	$(PYTHON) setup.py bdist_wheel upload

dist: clean
	$(PYTHON) setup.py sdist
	$(PYTHON) setup.py bdist_wheel
	ls -l dist

install: dist
	pip install --upgrade --no-deps --force-reinstall dist/$(PROJECT)-*.tar.gz

uninstall:
	pip uninstall -y $(PROJECT)

dist-deb:
	$(PYTHON) setup.py sdist
	rm -r debian/source
	$(PYTHON) setup.py --command-packages=stdeb.command debianize --with-python3 True
	$(PYTHON) setup.py prepare_debian_dist
	rename -f 's/$(PROJECT)-(.*)\.tar\.gz/$(PROJECT)_$$1\.orig\.tar\.gz/' dist/*.gz
	tar -xzf dist/$(PROJECT)_*.orig.tar.gz -C dist/
	cp -r debian dist/$(PROJECT)*/
	cd dist/$(PROJECT)*/; dpkg-source -b .
