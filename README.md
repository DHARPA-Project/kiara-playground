[![PyPI status](https://img.shields.io/pypi/status/kiara_modules.playground.svg)](https://pypi.python.org/pypi/kiara/)
[![PyPI version](https://img.shields.io/pypi/v/kiara_modules.playground.svg)](https://pypi.python.org/pypi/kiara/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/kiara_modules.playground.svg)](https://pypi.python.org/pypi/kiara/)
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2FDHARPA-Project%2Fkiara%2Fbadge%3Fref%3Ddevelop&style=flat)](https://actions-badge.atrox.dev/DHARPA-Project/kiara_modules.playground/goto?ref=develop)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# *kiara* modules (playground)

A set of commonly used/useful default modules (and pipelines) for [*Kiara*](https://github.com/DHARPA-project/kiara).

 - Documentation: [https://dharpa.org/kiara_modules.playground](https://dharpa.org/kiara_modules.playground)
 - Code: [https://github.com/DHARPA-Project/kiara_modules.playground](https://github.com/DHARPA-Project/kiara_modules.playground)

## Description

TODO

## Development

### Requirements

- Python (version >=3.6 -- some make targets only work for Python >=3.7 though)
- pip, virtualenv
- git
- make (on Linux / Mac OS X -- optional)


### Prepare development environment

If you only want to work on the modules, and not the core *Kiara* codebase, follow the instructions below. Otherwise, please
check the notes on how to setup a *Kiara* development environment under (TODO).

#### Linux & Mac OS X (using make)

For UNI*-like operating system, setting up a development environment is relatively easy:

```console
git clone https://github.com/DHARPA-Project/kiara_modules.playground.git
cd kiara_modules.playground
python3 -m venv .venv
source .venv/bin/activate
make init
```

#### Windows (or manual pip install)

It's impossible to lay out all the ways Python can be installed on a machine, and virtual- (or conda-)envs can be created, so I'll assume you know how to do this.
One simple way is to install the [Anaconda (individual edition)](https://docs.anaconda.com/anaconda/install/index.html), then use the Anaconda navigator to create a new environment, install the 'git' package in it (if your system does not already have it), and use the 'Open Terminal' option of that environment to start up a terminal that has that virtual-/conda-environment activated.

Once that is done, `cd` into a directory where you want this project folder to live, and do:

```console
# make sure your virtual env is activated!!!
git clone https://github.com/DHARPA-Project/kiara_modules.playground.git
cd kiara_modules.playground
pip install --extra-index-url https://pypi.fury.io/dharpa/ -U -e .[all_dev]
```

#### Try it out

After this is done, you should be able to run the included example module via:

```console
kiara run playground.sandbox.example text_1="xxx" text_2="yyy"
...
...
```

### Re-activate the development environment

The 'prepare' step from above only has to be done once. After that, to re-enable your virtual environment,
you'll need to navigate to the directory again (wherever that is, in your case), and run the ``source`` command from before again:

```console
cd path/to/kiara_modules.playground
source .venv/bin/activate  # if it isn't activated already, for example by the Anaconda navigator
kiara --help  # or whatever, point is, kiara should be available for you now,
```

### Enable development mode

*kiara* has a hidden mode that is designed to enable some convenience sub-commands that help when developing.

You can enable this mode by setting the ``DEVELOP`` environment variable to 'true':

```
export DEVELOP=true
```

Once this is done, *kiara* will display additional sub-commands when you use the ``--help`` flag. Currently, the only such
sub-command that is implemented is:

```
kiara data clear-data-store
```

This will delete all imported and generated data from the internal *kiara* data store.

### Update dependencies

In some cases the dependency *kiara* library or any of the packages that contain modules might have been updated, in which
case it is advisable to update the playground virtual-env. This can be done using:

```console
make update-dependencies
```

If you want to track the latest development version of those libraries, use the following instead:

```console
make update-dependencies-dev
```

*Note*: in order to revert back to a non-development version of the libraries, you'll have to remove the dependencies manually, and then re-install.


### ``make`` targets (Linux & Mac OS X)

- ``init``: init development project (install project & dev dependencies into virtualenv, as well as pre-commit git hook)
- ``update-dependencies``: update development dependencies (mainly the core ``kiara`` package from git and the main kiara module packages)
- ``flake``: run *flake8* tests
- ``mypy``: run mypy tests
- ``test``: run unit tests
- ``docs``: create static documentation pages (under ``build/site``)
- ``serve-docs``: serve documentation pages (incl. auto-reload) for getting direct feedback when working on documentation
- ``clean``: clean build directories

For details (and other, minor targets), check the ``Makefile``.


### Running tests

``` console
> make test
# or
> make coverage
```


## Copyright & license

This project is MPL v2.0 licensed, for the license text please check the [LICENSE](/LICENSE) file in this repository.

[Copyright (c) 2021 DHARPA project](https://dharpa.org)
