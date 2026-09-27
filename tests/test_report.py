"""One technical source, deterministic HTML, and immutable research artifacts."""

from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re

import pandas as pd
import pytest

from scripts import generate_report as report
from scripts.verify_release_artifacts import artifact_hashes, verify_document_values


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.headings, self.links, self.text = [], [], []
        self.tables = 0
        self.heading = False
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.heading = tag == "h2" or self.heading
        self.tables += tag == "table"
        if tag == "a":
            self.links.append(dict(attrs)["href"])

    def handle_endtag(self, tag):
        if tag == "h2":
            self.heading = False

    def handle_data(self, data):
        self.text.append(data)
        if self.heading:
            self.headings.append(data)


def test_html_matches_source_content_and_section_order():
    source = report.MARKDOWN_PATH.read_text(encoding="utf-8")
    html = report.HTML_PATH.read_text(encoding="utf-8")
    assert html == report.render_html(source)
    assert sha256(source.encode("utf-8")).hexdigest() in html
    document = Document(html)
    assert document.headings == re.findall(r"^## (.+)$", source, flags=re.MULTILINE)
    assert document.tables == 2
    for formula in re.findall(r"\\\[.*?\\\]|\\\(.*?\\\)", source, flags=re.DOTALL):
        assert formula in unescape(html)
    visible = re.sub(r"\s+", "", " ".join(document.text))
    for paragraph in source.split("\n\n"):
        if len(paragraph) < 80 or paragraph.startswith(("#", "|", "-", "```", "\\[")):
            continue
        plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", paragraph)
        plain = plain.replace("**", "").replace("`", "")
        assert re.sub(r"\s+", "", plain) in visible
    for target in document.links:
        if not target.startswith(("http:", "https:", "#")):
            assert (report.HTML_PATH.parent / target).resolve().is_file()


def test_generation_does_not_touch_research_and_reads_live_source(tmp_path, monkeypatch):
    before = artifact_hashes()
    generated = tmp_path / "report.html"
    report.generate_report(generated)
    assert generated.read_bytes() == report.HTML_PATH.read_bytes()
    assert artifact_hashes() == before
    source = tmp_path / "methodology.md"
    source.write_text("# Example\n\n## First\n\nCurrent source.\n", encoding="utf-8")
    monkeypatch.setattr(report, "MARKDOWN_PATH", source)
    report.generate_report(generated)
    assert "Current source." in generated.read_text(encoding="utf-8")
    source.write_text("# Example\n\n## Second\n\nChanged source.\n", encoding="utf-8")
    report.generate_report(generated)
    assert "Changed source." in generated.read_text(encoding="utf-8")
    assert "Current source." not in generated.read_text(encoding="utf-8")


def test_renderer_escapes_html_and_preserves_code_math_and_tables():
    source = "# Test\n\n<script>bad()</script>\n\n```text\nA -> B\n```\n\n"
    source += r"Inline \(x=-1\), **bold**, `code`." + "\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    html = report.render_html(source)
    assert "<script>bad()" not in html
    assert "&lt;script&gt;bad()&lt;/script&gt;" in html
    assert "<strong>bold</strong>" in html and "<code>code</code>" in html
    assert r"\(x=-1\)" in html
    assert "<pre><code>A -&gt; B</code></pre>" in html
    assert "<td>1</td><td>2</td>" in html
    with pytest.raises(ValueError, match="Unclosed"):
        report.render_html("# Test\n\n```text\nmissing end")


def test_verifier_rejects_changed_canonical_or_research_values():
    from types import SimpleNamespace as Result

    source = report.MARKDOWN_PATH.read_text(encoding="utf-8")
    data = {
        "type_a": Result(summary=Result(conversions=3069.9809275260704, cpa=1_200_000 / 3069.9809275260704)),
        "calendar_type_a": Result(summary=Result(conversions=3077.0820480462485, cpa=1_200_000 / 3077.0820480462485)),
        "type_b": Result(summary=Result(spend=104138.52082512155)),
        "infeasible": Result(feasibility=Result(maximum_achievable=3952.4774377056415,
                                               reason=Result(value="MARKET_CAPACITY"))),
        "adaptive_summary": pd.read_csv(report.REPO_ROOT / "experiments" / "results" / "summary.csv"),
    }
    verify_document_values(source, data)
    for value in ("3 069,98", "3 077,08", "104 138,52", "3 952,48", "3 168,35"):
        with pytest.raises(AssertionError):
            verify_document_values(source.replace(value, "0,00"), data)
