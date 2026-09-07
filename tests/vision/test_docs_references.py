"""Every ``docs/*.md`` pointer in the source must name a section, and that section must exist.

Docstrings that promise documentation ("recorded in ``docs/vision.md``") outlive the paragraph
they were written against: the doc gets restructured, the promised list is never written, and
nothing fails. So a pointer has to name its target -- ``"Data leakage" in docs/vision.md`` --
and this test resolves that name against the file's actual headings.
"""

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCANNED_TREES = ("src/pyclad", "examples")

# A pointer that names its section: `"Some heading" in docs/whatever.md`, with optional
# reST/Markdown backticks around the path. Whitespace is normalised first, so the phrase is
# still matched when a docstring wraps it across lines.
NAMED_REFERENCE = re.compile(
    r"[\"'`]{1,2}(?P<section>[^\"'`\n]+?)[\"'`]{1,2}\s+in\s+`{0,2}(?P<doc>docs/[\w./-]+\.md)`{0,2}"
)
ANY_REFERENCE = re.compile(r"docs/[\w./-]+\.md")
MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$", re.MULTILINE)
# Headings carry inline markup (`**bold**`, `` `code` ``) that a prose pointer will not repeat.
HEADING_MARKUP = re.compile(r"[*_`]")


def _python_sources():
    for tree in SCANNED_TREES:
        for path in sorted((REPO_ROOT / tree).rglob("*.py")):
            yield path


def _normalised(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _headings_of(doc: pathlib.Path) -> set:
    return {
        HEADING_MARKUP.sub("", m.group("title")).strip().casefold()
        for m in MARKDOWN_HEADING.finditer(doc.read_text(encoding="utf-8"))
    }


def test_every_docs_pointer_names_a_section():
    unnamed = []
    for path in _python_sources():
        text = _normalised(path.read_text(encoding="utf-8"))
        named_spans = [m.span("doc") for m in NAMED_REFERENCE.finditer(text)]
        for match in ANY_REFERENCE.finditer(text):
            if not any(start <= match.start() and match.end() <= end for start, end in named_spans):
                unnamed.append(f"{path.relative_to(REPO_ROOT)}: {match.group()}")

    assert not unnamed, (
        "These pointers name a doc file but no section inside it, so nothing can verify the "
        "promised content exists. Write them as '\"Section heading\" in docs/<file>.md':\n  " + "\n  ".join(unnamed)
    )


def test_every_referenced_section_exists():
    missing = []
    for path in _python_sources():
        for match in NAMED_REFERENCE.finditer(_normalised(path.read_text(encoding="utf-8"))):
            doc = REPO_ROOT / match.group("doc")
            section = match.group("section").strip()
            if not doc.is_file():
                missing.append(f"{path.relative_to(REPO_ROOT)}: {match.group('doc')} does not exist")
            elif section.casefold() not in _headings_of(doc):
                missing.append(f"{path.relative_to(REPO_ROOT)}: {match.group('doc')} has no heading {section!r}")

    assert not missing, "Referenced documentation sections are missing:\n  " + "\n  ".join(missing)


def test_the_check_would_catch_a_broken_pointer(tmp_path):
    """Guard against the regexes silently matching nothing and the tests above passing vacuously."""
    text = _normalised('promised under "Nonexistent Heading" in ``docs/vision.md``.')
    match = NAMED_REFERENCE.search(text)
    assert match is not None and match.group("doc") == "docs/vision.md"
    assert match.group("section") == "Nonexistent Heading"
    assert match.group("section").casefold() not in _headings_of(REPO_ROOT / "docs/vision.md")
