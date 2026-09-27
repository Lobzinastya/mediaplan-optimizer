"""Read-only presentation of the methodology Markdown; no planning state."""

from pathlib import Path
import re

import streamlit as st


DOCUMENT_PATH = Path(__file__).resolve().parent / "docs" / "MATHEMATICAL_MODEL.md"


def load_methodology(path: Path = DOCUMENT_PATH) -> tuple[str, str, list[tuple[str, str]]]:
    """Read the title, introduction and level-two sections on every visit."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^## (.+)$", text, flags=re.MULTILINE)
    title, introduction = parts[0].strip().split("\n", 1)
    if not title.startswith("# ") or len(parts) < 3:
        raise ValueError(f"Methodology must have a title and level-two sections: {path}")
    sections = [(parts[index].strip(), parts[index + 1].strip())
                for index in range(1, len(parts), 2)]
    return title[2:], introduction.strip(), sections


def streamlit_markdown(text: str) -> str:
    """Adapt the source's LaTeX delimiters to Streamlit's native math syntax."""
    text = re.sub(r"\\\[(.*?)\\\]", lambda match: "\n$$\n" + match[1].strip() + "\n$$\n",
                  text, flags=re.DOTALL)
    return re.sub(r"\\\((.*?)\\\)", lambda match: "$" + match[1] + "$", text)


def render_methodology() -> None:
    title, introduction, sections = load_methodology()
    st.title(title)
    st.markdown(streamlit_markdown(introduction))
    for heading, body in sections:
        with st.expander(heading, expanded=False):
            # Local repository references become downloads, not broken app URLs.
            references = re.findall(r"\[([^\]]+)\]\((\.\./[^)]+)\)", body)
            for label, target in references:
                body = body.replace(f"[{label}]({target})", label)
            st.markdown(streamlit_markdown(body))
            for label, target in references:
                path = (DOCUMENT_PATH.parent / target).resolve()
                path.relative_to(DOCUMENT_PATH.parent.parent)
                st.download_button(label, path.read_bytes(), file_name=path.name,
                                   key=f"methodology:{heading}:{target}", on_click="ignore")
