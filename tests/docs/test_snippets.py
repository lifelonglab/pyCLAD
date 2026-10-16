"""The code shown in the docs runs, so the docs cannot drift from the library unnoticed.

Pages show code by including files from ``docs/snippets/`` (``--8<-- "docs/snippets/..."``), and every
snippet is run here. A snippet may start with ``# requires: torch, ...`` to be skipped when an optional
package is missing. Snippets under ``docs/snippets/longrun/`` download data, so by default only their
pyCLAD imports are checked; ``--longrun`` runs them. Snippets starting with ``# needs local data: ...`` read data
that has to be obtained by hand (for example image benchmarks), so they are never run and only their imports are
checked.

Python blocks written directly in a page are limited to a few lines, and the README's quick-start, which
GitHub cannot render from a snippet, is run as well.
"""

import ast
import importlib
import pathlib
import re
import runpy

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
SNIPPETS = DOCS / "snippets"
LONGRUN_SNIPPETS = SNIPPETS / "longrun"

# Enough for a signature or a one-line call, too short for an example that could go stale.
MAX_INLINE_PYTHON_LINES = 3

REQUIRES = re.compile(r"^# requires: (?P<modules>.+)$", re.MULTILINE)
NEEDS_LOCAL_DATA = re.compile(r"^# needs local data: .+$", re.MULTILINE)
FENCED_PYTHON = re.compile(r"^(?P<indent>[ \t]*)```(?:python|py)\b[^\n]*\n(?P<body>.*?)^(?P=indent)```", re.S | re.M)
SNIPPET_INCLUDE = re.compile(r'^--8<-- "(?P<path>[^":]+)(?::[\w-]+)?"$')


def _all_snippets():
    return sorted(SNIPPETS.rglob("*.py"))


def _is_longrun(path: pathlib.Path) -> bool:
    return LONGRUN_SNIPPETS in path.parents


def _needs_local_data(path: pathlib.Path) -> bool:
    return NEEDS_LOCAL_DATA.search(path.read_text(encoding="utf-8")) is not None


def _runnable():
    return [path for path in _all_snippets() if not _is_longrun(path) and not _needs_local_data(path)]


def _longrun():
    return [path for path in _all_snippets() if _is_longrun(path) and not _needs_local_data(path)]


def _not_run_by_default():
    return [path for path in _all_snippets() if _is_longrun(path) or _needs_local_data(path)]


def _id(path: pathlib.Path) -> str:
    return str(path.relative_to(SNIPPETS))


def _skip_unless_installed(snippet: pathlib.Path):
    match = REQUIRES.search(snippet.read_text(encoding="utf-8"))
    if match:
        for module in match.group("modules").split(","):
            pytest.importorskip(module.strip())


def _run(path: pathlib.Path, tmp_path: pathlib.Path, monkeypatch):
    # Run where outputs (output.json, plots) can be written without touching the repository.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    runpy.run_path(str(path), run_name="__main__")


@pytest.mark.parametrize("snippet", _runnable(), ids=_id)
def test_snippet_runs(snippet, tmp_path, monkeypatch):
    _skip_unless_installed(snippet)
    _run(snippet, tmp_path, monkeypatch)


@pytest.mark.longrun
@pytest.mark.parametrize("snippet", _longrun(), ids=_id)
def test_longrun_snippet_runs(snippet, tmp_path, monkeypatch):
    _skip_unless_installed(snippet)
    _run(snippet, tmp_path, monkeypatch)


@pytest.mark.parametrize("snippet", _not_run_by_default(), ids=_id)
def test_snippet_imports_exist(snippet):
    """The part of a snippet that a rename would break, checked without its data."""
    _skip_unless_installed(snippet)
    tree = ast.parse(snippet.read_text(encoding="utf-8"), filename=str(snippet))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "pyclad":
            module = importlib.import_module(node.module)
            missing = [alias.name for alias in node.names if not hasattr(module, alias.name)]
            assert not missing, f"{_id(snippet)}: {node.module} has no {missing}"


def _pages():
    return sorted(DOCS.rglob("*.md"))


def test_every_snippet_is_shown_in_the_docs():
    included = set()
    for page in _pages():
        for block in FENCED_PYTHON.finditer(page.read_text(encoding="utf-8")):
            for line in block.group("body").splitlines():
                match = SNIPPET_INCLUDE.match(line.strip())
                if match:
                    included.add((REPO_ROOT / match.group("path")).resolve())
    unused = [_id(path) for path in _all_snippets() if path.resolve() not in included]
    assert not unused, f"Snippets no page includes: {unused}"


def test_pages_show_long_code_only_through_snippets():
    offenders = []
    for page in _pages():
        for block in FENCED_PYTHON.finditer(page.read_text(encoding="utf-8")):
            lines = [line.strip() for line in block.group("body").splitlines() if line.strip()]
            if all(SNIPPET_INCLUDE.match(line) for line in lines):
                continue
            if len(lines) > MAX_INLINE_PYTHON_LINES:
                offenders.append(f"{page.relative_to(REPO_ROOT)}: block starting with {lines[0]!r}")
    assert not offenders, (
        f"Python blocks longer than {MAX_INLINE_PYTHON_LINES} lines must come from a file in docs/snippets/, "
        "so that this test runs them:\n  " + "\n  ".join(offenders)
    )


def test_readme_quickstart_runs(tmp_path, monkeypatch):
    blocks = [m.group("body") for m in FENCED_PYTHON.finditer((REPO_ROOT / "README.md").read_text(encoding="utf-8"))]
    assert blocks, "README.md has no python block; update this test if the quick-start moved"
    script = tmp_path / "readme_quickstart.py"
    script.write_text("\n".join(blocks), encoding="utf-8")
    _run(script, tmp_path, monkeypatch)
