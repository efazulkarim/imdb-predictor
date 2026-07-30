"""Compatibility wrapper for the Strict OOXML conference template."""

from __future__ import annotations

import zipfile
from pathlib import Path

import build_ieee_conference_paper as manuscript


def strict_template_to_transitional(source: Path, target: Path) -> None:
    replacements = {
        b"http://purl.oclc.org/ooxml/wordprocessingml/main": b"http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        b"http://purl.oclc.org/ooxml/officeDocument/relationships": b"http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        b"http://purl.oclc.org/ooxml/drawingml/main": b"http://schemas.openxmlformats.org/drawingml/2006/main",
        b"application/vnd.ms-word.document.main+xml": b"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
        b"application/vnd.ms-word.styles+xml": b"application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml",
        b"application/vnd.ms-word.fontTable+xml": b"application/vnd.openxmlformats-officedocument.wordprocessingml.fontTable+xml",
        b"application/vnd.ms-word.settings+xml": b"application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml",
        b"application/vnd.ms-word.numbering+xml": b"application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml",
        b"application/vnd.ms-word.footnotes+xml": b"application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml",
        b"application/vnd.ms-word.endnotes+xml": b"application/vnd.openxmlformats-officedocument.wordprocessingml.endnotes+xml",
    }
    with zipfile.ZipFile(source, "r") as src, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as dst:
        for member in src.infolist():
            data = src.read(member.filename)
            if member.filename.endswith((".xml", ".rels")):
                for old, new in replacements.items():
                    data = data.replace(old, new)
            dst.writestr(member, data)


manuscript.strict_template_to_transitional = strict_template_to_transitional
manuscript.build()
