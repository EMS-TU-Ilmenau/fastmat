# Contribution guidelines

You are welcome to participate in this project and contribute new features, bug fixes or enhancements as you see fit.

## How this repository is organized
The master branch is protected and cannot be pushed or merged into directly. Therefore you will need to do your work on a dedicated branch and issue a merge request into master when finished. There are automatic checks in place that check every merge request for fails oder major violations to prevent

## Things to do before issuing a pull request
1. Test it thoroughly
2. Make sure your commits are meaningful in message, description and partitioning. Rebase if neccessary. Check out this page if you feel uncomfortable: https://chris.beams.io/posts/git-commit/
3. Have a look at the diff log over the whole branch to check once more you do not introduce regressions (missed merge conflicts, accidentially removed lines, stuff like that). These kind of things occur very easily and should not be taken lightly.
4. When you feel comfortable, run the following commands as the final check routine:
    ```
    git clean -dxf
    make styleCheck
    make test PYTHON=python2
    make test PYTHON=python3
    ```
   Fix any errors and test issues that get revealed here
5. when everything finishes without complaints, open up a pull request to master
