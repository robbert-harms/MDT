PYTHON=$$(which python3)
PIP=$$(which pip3)
PROJECT_NAME=mdt
PROJECT_VERSION=$$($(PYTHON) setup.py --version)
GPG_SIGN_KEY=0E1AA560
UBUNTU_MAIN_TARGET_DISTRIBUTIONS=xenial
UBUNTU_OTHER_TARGET_DISTRIBUTIONS=bionic


.PHONY: help
help:
	@echo "clean - remove all build, test, coverage and Python artifacts (no uninstall)"
	@echo "lint - check style with flake8"
	@echo "test(s)- run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "docs-pdf - generate the PDF documentation, including API docs"
	@echo "docs-man - generate the linux manpages"
	@echo "docs-changelog - generate the changelog documentation"
	@echo "prepare-release - prepare for a new release"
	@echo "release - package and upload a release"
	@echo "dist - create a pip package"
	@echo "dist-ubuntu - create an Ubuntu package"
	@echo "install - installs the package using pip"
	@echo "uninstall - uninstalls the package using pip"

.PHONY: clean
clean: clean-build clean-pyc clean-test
	$(PYTHON) setup.py clean

.PHONY: clean-build
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-test
clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

.PHONY: lint
lint:
	flake8 $(PROJECT_NAME) tests

.PHONY: test
test:
	$(PYTHON) setup.py test

.PHONY: tests
tests: test

.PHONY: test-all
test-all:
	tox

.PHONY: coverage
coverage:
	coverage run --source $(PROJECT_NAME) setup.py test
	coverage report -m
	coverage html
	@echo "To view results type: htmlcov/index.html &"

.PHONY: docs
docs:
	rm -f docs/$(PROJECT_NAME)*.rst
	rm -f docs/modules.rst
	$(MAKE) -C docs clean
	sphinx-apidoc -o docs/ $(PROJECT_NAME)
	$(MAKE) -C docs html SPHINXBUILD='python3 $(shell which sphinx-build)'
	@echo "To view results type: firefox docs/_build/html/index.html &"

.PHONY: docs-pdf
docs-pdf:
	rm -f docs/$(PROJECT_NAME)*.rst
	rm -f docs/modules.rst
	$(MAKE) -C docs clean
	sphinx-apidoc -o docs/ $(PROJECT_NAME)
	$(MAKE) -C docs latexpdf SPHINXBUILD='python3 $(shell which sphinx-build)'
	@echo "To view results use something like: evince docs/_build/latex/$(PROJECT_NAME).pdf &"

.PHONY: docs-man
docs-man:
	rm -f docs/$(PROJECT_NAME)*.rst
	rm -f docs/modules.rst
	$(MAKE) -C docs clean
	sphinx-apidoc -o docs/ $(PROJECT_NAME)
	$(MAKE) -C docs man SPHINXBUILD='python3 $(shell which sphinx-build)'
	@echo "To view results use something like: man docs/_build/man/$(PROJECT_NAME).1 &"

.PHONY: docs-changelog
docs-changelog:
	gitchangelog

.PHONY: prepare-release
prepare-release: clean
	@echo "Current version: "$(PROJECT_VERSION)
	@while [ -z "$$NEW_VERSION" ]; do \
        read -r -p "Give new version: " NEW_VERSION;\
    done && \
    ( \
        printf 'Setting new version: %s \n\n' \
        	"$$NEW_VERSION " \
	) && sed -i -e "s/^\(VERSION\ =\ \)\('.*'\)\(\ *\)/\1'$$NEW_VERSION'/g" $(PROJECT_NAME)/__version__.py
	$(MAKE) docs-changelog
	@echo "Consider manually inspecting CHANGELOG.rst for possible improvements."

# todo: add GitHub Releases API hook here
.PHONY: release
release: clean release-ubuntu-ppa release-pip release-github

.PHONY: release-pip
release-pip:
	$(PYTHON) setup.py sdist bdist_wheel
	twine upload dist/*.{whl,tar.gz}

.PHONY: release-ubuntu-ppa
release-ubuntu-ppa: dist-ubuntu
	dput ppa:robbert-harms/cbclab dist/$(PROJECT_NAME)_$(PROJECT_VERSION)-1_source.changes
	for ubuntu_version in $(UBUNTU_OTHER_TARGET_DISTRIBUTIONS) ; do \
		dput ppa:robbert-harms/cbclab dist/$(PROJECT_NAME)_$(PROJECT_VERSION)-1~$${ubuntu_version}1_source.changes ; \
	done

.PHONY: release-github
release-github:
	git push . master:latest_release
	git tag -a v$(PROJECT_VERSION) -m "Version $(PROJECT_VERSION)"
	git push origin latest_release
	git push origin --tags

.PHONY: dist
dist: clean
	$(PYTHON) setup.py sdist
	$(PYTHON) setup.py bdist_wheel
	ls -l dist

.PHONY: dist-ubuntu
dist-ubuntu: clean
	$(PYTHON) setup.py sdist
	cp dist/$(PROJECT_NAME)-$(PROJECT_VERSION).tar.gz dist/$(PROJECT_NAME)_$(PROJECT_VERSION).orig.tar.gz
	tar -xzf dist/$(PROJECT_NAME)-$(PROJECT_VERSION).tar.gz -C dist/
	$(MAKE) _package-ubuntu suite=$(UBUNTU_MAIN_TARGET_DISTRIBUTIONS) debian-version=1 build-flag=-sa

	for ubuntu_version in $(UBUNTU_OTHER_TARGET_DISTRIBUTIONS) ; do \
		$(MAKE) _package-ubuntu suite=$$ubuntu_version debian-version=1~$${ubuntu_version}1 build-flag=-sd ; \
	done

.PHONY: _package-ubuntu
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

.PHONY: install
install: dist
	$(PIP) install --upgrade --no-deps --force-reinstall dist/$(PROJECT_NAME)-*.tar.gz

.PHONY: uninstall
uninstall:
	$(PIP) uninstall -y $(PROJECT_NAME)
