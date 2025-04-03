# Contributing to pyCLAD

:+1: Thanks for taking the time to contribute! :+1:

# Code of Conduct

Please make sure that you:

- Follow the standard python naming conventions.
- Follow the naming conventions of the project.
- Write clean and readable code.
- Write docstrings for all functions and classes.
- Write unit tests for all functions and classes.
- Write an informative commit message.
- Divide your code into commits that are small and focused on a single task.
- Write a brief description of the changes in the pull request.
- Make sure that the code passes all the static code analysis checks.
- Make sure that the code passes all the unit tests.
- Make sure that the code is compatible with Python 3.11 and above.

## Static code analysis

To ensure the quality of the project, we leverage a few static code analysis tools listed below.
Please make sure that your code passes these checks before submitting a pull request.

You can install all these tools by running the following command:
```pip install black flake8 isort```

### Black

Black is a tool that automatically formats Python code. You can run it by executing the following command in the root
directory of the project.
```black src```

If you just want to check if the code is formatted correctly without `black` doing any changes, you can run the
following command:
```black --diff src```

See more info about black [here](https://black.readthedocs.io/en/stable/)

### Flake8

Flake8 is a tool that checks the style and quality of code.
You can run it by executing the following command in the root directory of the project:
```flake8 src```

See more info about flake8 [here](https://flake8.pycqa.org/en/latest/)

### isort

isort is a tool that sorts imports ensuring unified import style across the project.
You can run it by executing the following command in the root directory of the project:
```isort src```

If you just want to check if the code is formatted correctly without `isort` doing any changes, you can run the
following command:
```isort src --diff```

See more info about isort [here](https://pycqa.github.io/isort/)

## Unit tests

Unit tests are essential tool to ensure the reliability of the software.
This project leverages `pytest` for running unit tests. You can install it by running the following command:
```pip install pytest```

You can run the unit tests by executing the following command in the root directory of the project:
```pytest tests```

Please make sure that:

* You write unit tests for your new functions and classes.
* You execute all the existing unit tests to ensure that your changes did not break anything.

Some tests may take a long time to execute. They should be annotated with `@pytest.mark.longrun` decorator. They are
disabled by default. If you want to include them in tests execution, you should use the following command:
```pytest tests --longrun```

# Documentation

Please make sure that you update the documentation if you are adding new features or changing existing ones.
We use MKDocs for documentation.
You have to install additional dependencies listed in `docs/requirements.txt` to build the documentation.

All our documentation files are stored in the `docs` directory, while config file `mkdocs.yml` is in the root directory
of the project.

You can see the live preview of the documentation while working on it by running the following command:
```mkdocs serve```

You can find more info about MKDocs [here](https://www.mkdocs.org/) and MKDocs Material
theme [here](https://squidfunk.github.io/mkdocs-material/)
