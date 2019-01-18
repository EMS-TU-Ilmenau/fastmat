# Contribution Guidelines

You are welcome to participate in this project and contribute new features, bug fixes or enhancements as you see fit.

## Issues
We love issues! Especially those that do not exist. But if you have found one in the sense that you think there is a bug, a feature missing or something wrong with something else, we invite you to open them plentyfully and explain what is wrong, slow or disturbing in any way.

## How this repository is organized
The master branch is protected and cannot be pushed or merged into directly. Therefore you will need to do your work on a dedicated branch and issue a pull request into master when finished. There are automatic checks in place that check every merge request for code styles and errors within the classes.

We aim at a subset of the [PEP8](https://www.python.org/dev/peps/pep-0008/) style guide, where we have removed the checks for E26, E116, E203, E221, E222, E225, E227, E241, E402, E731 and W504.

Any commit in master will start a process of thorough testing through our CI. First, automated builds and tests will be started for Python 2, Python3 and Python3 on OS X. Then, deeper code analysis is triggered (coveralls, codacy, ...) and the results get published by these tools. Finally, if no errors were found and everything *just works*, `master` will be merged automatically into `stable`. This way we do know that `stable` is always up-to-date and working.

The release of fastmat versions *into the wild* is triggered by defining a fresh version tag into the file `.version`. This then gets processed by setup.py and controls the fastmat version number used for deployment and documentation. Pypi assists us by only accepting the first build of a version. Therefore only when we update the `.version`file, a new version gets released. If and only if all wheels were built the tag specifying the release version is set for the newest commit in the `stable` branch (being the released state).

## Pull Requests
We would like to organize our work by tracking ongoing work in pull requests. So as soon as you have started hacking away in a new branch, you can push it upstream and issue a pull request to get the discussion with everyone else going. If it is a larger feature or a more involved refactoring, there might be some discussion happening and we might need some iterations and modifications in your branch until we see it fit for merging.

Since we aim at providing some simple checks of your code whenever pushing to some feature branch, the code itself should already be of high(ish) quality if these checks pass in the sense of coding style and test approval.

Feel free to open any issues and associate them to your pull request in order to allow a more elaborate and detailed discussion of things there and also to keep everything organized and transparent.
