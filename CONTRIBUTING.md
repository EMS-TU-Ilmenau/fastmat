# Contribution Guidelines

You are welcome to participate in this project and contribute new features, bug fixes or enhancements as you see fit.

## Issues
We love issues! Especially those that do not exist. But if you have found one in the sense that you think there is a bug, a feature missing or something wrong with something else, we invite you to open them plentyfully and explain what is wrong, slow or disturbing in any way.

## How this repository is organized
The master branch is protected and cannot be pushed or merged into directly. Therefore you will need to do your work on a dedicated branch and issue a pull request into master when finished. There are automatic checks in place that check every merge request for code styles and errors within the classes. the holy grail in this repository is `stable`, which serves as the branch where we issue new realeses from and our build servers compile and deploy the packages automatically to `pip`. Occasionally we see the current state in `master` fit and we merge into stable, which then makes hell break loose and pushes out a new version of fastmat into the wild.

## Pull Requests
We would like to organize our work by tracking ongoing work in pull reqeusts. So as soon as you have started hacking away in a new branch, you can push it upstream and issue a pull request to get the discussion with everyone else going going. If it is a larger feature or a more involved refactoring, there ight be some discussion happening and we might need some iterations and modifications in your branch until we see it fit for merging.

Since we aim at providing some simple checks of your code whenever pushing to some feature branch, the code itself should already be of high(ish) quality if these checks pass in the sense of coding style and test approval.

Feel free to open any issues and associate them to your pull request in order to allow a more elaborate and detailed discussion of things there and also to keep everything organized and transparent.
