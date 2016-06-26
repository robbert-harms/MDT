.PHONY: clean clean-build clean-pyc clean-test clean-qt-designs lint test tests test-all coverage docs qt-designs release dist install uninstall

help:
	@echo "clean - remove all build, test, coverage and Python artifacts (no uninstall)"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "clean-qt-designs - clean the build QT designs"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "tests - synonym for test"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "qt-designs - build all the QT designs and objects"
	@echo "release - package and upload a release"
	@echo "dist - create the package"
	@echo "install - installs the package using pip"
	@echo "uninstall - uninstalls the package using pip"
	
clean: clean-build clean-pyc clean-test clean-qt-designs

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

clean-qt-designs:
	$(MAKE) -C mdt/data/qt_designs/ clean	

lint:
	flake8 mdt tests

test:
	python setup.py test

tests: test

test-all:
	tox

coverage:
	coverage run --source mdt setup.py test
	coverage report -m
	coverage html
	@echo "To view results type: htmlcov/index.html &"

docs:
	rm -f docs/mdt.rst
	rm -f docs/modules.rst
	sphinx-apidoc -f -o docs/ mdt
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	@echo "To view results type: firefox docs/_build/html/index.html &"

qt-designs:
	$(MAKE) -C mdt/data/qt_designs/ all

release: clean qt-designs
	python setup.py sdist upload
	python setup.py bdist_wheel upload

dist: clean qt-designs
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: dist
	pip install --upgrade --no-deps --force-reinstall dist/mdt-*.tar.gz

uninstall: 
	pip uninstall -y mdt
