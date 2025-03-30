def pytest_addoption(parser):
    parser.addoption(
        "--longrun", action="store_true", dest="longrun", default=False, help="enable longrun decorated tests"
    )


def pytest_configure(config):
    if not config.option.longrun:
        setattr(config.option, "markexpr", "not longrun")
