"""Render the single methodology source as HTML, without running experiments."""

from __future__ import annotations

import hashlib
import html
import os
from pathlib import Path
import re
from urllib.parse import urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_PATH = REPO_ROOT / "docs" / "MATHEMATICAL_MODEL.md"
HTML_PATH = REPO_ROOT / "report" / "report.html"


def _inline(text: str) -> str:
    """Render the inline syntax used in the report; leave TeX intact."""
    pattern = r"\\\(.*?\\\)|`[^`]+`|\*\*.+?\*\*|\[[^\]]+\]\([^)]+\)"
    result, previous = [], 0
    for match in re.finditer(pattern, text):
        result.append(html.escape(text[previous:match.start()]))
        token = match.group()
        if token.startswith(r"\("):
            result.append(html.escape(token))
        elif token.startswith("`"):
            result.append("<code>" + html.escape(token[1:-1]) + "</code>")
        elif token.startswith("**"):
            result.append("<strong>" + _inline(token[2:-2]) + "</strong>")
        else:
            label, target = token[1:-1].split("](", 1)
            parsed = urlsplit(target)
            if parsed.scheme and parsed.scheme not in {"https", "http", "mailto"}:
                raise ValueError(f"Unsupported link scheme: {target}")
            if not parsed.scheme and parsed.path:
                local = (MARKDOWN_PATH.parent / parsed.path).resolve()
                target = Path(os.path.relpath(local, HTML_PATH.parent)).as_posix()
                if parsed.fragment:
                    target += "#" + parsed.fragment
            result.append(f'<a href="{html.escape(target, quote=True)}">{_inline(label)}</a>')
        previous = match.end()
    result.append(html.escape(text[previous:]))
    return "".join(result)


def markdown_body(markdown: str) -> str:
    """Convert the report's headings, paragraphs, lists, tables, code and TeX.

    This small, dependency-free renderer supports the syntax used by this
    document, not arbitrary CommonMark. Raw HTML is escaped.
    """
    lines = markdown.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("```") or line == r"\[":
            is_code = line.startswith("```")
            closing = "```" if is_code else r"\]"
            start = index
            index += 1
            while index < len(lines) and lines[index].strip() != closing:
                index += 1
            if index == len(lines):
                raise ValueError(f"Unclosed block at line {start + 1}")
            text = "\n".join(lines[start + 1:index])
            blocks.append("<pre><code>" + html.escape(text) + "</code></pre>" if is_code
                          else '<div class="math">' + html.escape("\\[\n" + text + "\n\\]") + "</div>")
            index += 1
        elif heading := re.match(r"^(#{1,6}) (.+)$", line):
            level = len(heading[1])
            blocks.append(f"<h{level}>{_inline(heading[2])}</h{level}>")
            index += 1
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
                index += 1
            if len(rows) < 2 or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1]):
                raise ValueError("Table is missing its Markdown separator row")
            if any(len(row) != len(rows[0]) for row in rows):
                raise ValueError("Inconsistent Markdown table width")
            head = "".join(f"<th>{_inline(cell)}</th>" for cell in rows[0])
            body = "".join("<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in row) + "</tr>"
                           for row in rows[2:])
            blocks.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        elif line.startswith("- "):
            items = []
            while index < len(lines) and lines[index].strip().startswith("- "):
                items.append("<li>" + _inline(lines[index].strip()[2:]) + "</li>")
                index += 1
            blocks.append("<ul>" + "".join(items) + "</ul>")
        else:
            paragraph = []
            while index < len(lines) and lines[index].strip():
                paragraph.append(lines[index].strip())
                index += 1
            blocks.append("<p>" + _inline(" ".join(paragraph)) + "</p>")
    return "\n\n".join(blocks) + "\n"


def render_html(markdown: str) -> str:
    title = next((line[2:] for line in markdown.splitlines() if line.startswith("# ")), None)
    if not title:
        raise ValueError("The methodology needs a level-one title")
    digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    head = (
        '<!doctype html>\n<html lang="ru"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta name="source-sha256" content="{digest}">\n'
        f"<title>{html.escape(title)}</title>\n"
        "<style>body{max-width:1000px;margin:3rem auto;padding:0 1rem;"
        "font:16px/1.6 system-ui,sans-serif;color:#17202a}"
        "h1,h2{color:#0f766e}table{border-collapse:collapse;width:100%;font-size:.9rem}"
        "th,td{border:1px solid #ccd1d1;padding:.45rem;text-align:left}"
        "th{background:#f2f4f4}pre{overflow-x:auto;padding:1rem;background:#f2f4f4}"
        ".math{overflow-x:auto}a{color:#0f766e}"
        "@media print{body{margin:0;max-width:none}h2{break-after:avoid}}</style>\n"
        '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>\n'
        "</head><body>\n<main>\n"
    )
    return head + markdown_body(markdown) + "</main>\n</body></html>\n"


def generate_report(output_path: Path = HTML_PATH) -> None:
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(markdown), encoding="utf-8", newline="\n")


def main() -> None:
    generate_report()
    print(f"Generated {HTML_PATH.relative_to(REPO_ROOT)} from {MARKDOWN_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
