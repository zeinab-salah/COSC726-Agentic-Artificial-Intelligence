"""Build COSC726_Project_Zeinab_Salah_Report.pdf from REPORT.md. Standard fonts only."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "REPORT.md"
OUTPUT = ROOT / "COSC726_Project_Zeinab_Salah_Report.pdf"
FONT = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"


class ReportPDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_x(self.l_margin)
        self.set_font("Body", size=9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, "COSC726  |  Zeinab Salah  |  Grounded technical briefs", align="C")
        self.ln(6)
        self.set_x(self.l_margin)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Body", size=9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, str(self.page_no()), align="C")
        self.set_text_color(0, 0, 0)

    def block(self, height: float, text: str) -> None:
        self.multi_cell(0, height, text, new_x="LMARGIN", new_y="NEXT")


def wrap_code(text: str) -> str:
    text = text.replace("`", "")
    return re.sub(r"(\S{42})", r"\1 ", text)


def build() -> None:
    if not SOURCE.exists():
        raise SystemExit("REPORT.md is missing. Run python -m brief.run_demo first.")
    pdf = ReportPDF(format="A4")
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_font("Body", "", FONT)
    pdf.add_font("Body", "B", FONT_BOLD)
    pdf.add_page()
    pdf.set_font("Body", "B", 18)
    pdf.block(9, "Grounded technical briefs")
    pdf.set_font("Body", "", 12)
    pdf.ln(4)
    pdf.block(7, "A bounded three-role crew for citation-grounded writing")
    pdf.ln(6)
    pdf.set_font("Body", "B", 12)
    pdf.block(7, "COSC726 Agentic Artificial Intelligence")
    pdf.set_font("Body", "", 12)
    pdf.block(7, "Student: Zeinab Salah")
    pdf.ln(4)
    meta = {}
    body_lines = SOURCE.read_text(encoding="utf-8").splitlines()
    for line in body_lines:
        if line.startswith("Repository URL:"):
            meta["repo"] = line.split(":", 1)[1].strip()
        if line.startswith("Video URL"):
            meta["video"] = line.split(":", 1)[1].strip()
    pdf.set_font("Body", "B", 11)
    pdf.block(7, "Repository URL")
    pdf.set_font("Body", "", 11)
    pdf.block(7, meta.get("repo", "REPLACE_WITH_GITHUB_URL"))
    pdf.ln(2)
    pdf.set_font("Body", "B", 11)
    pdf.block(7, "Video URL (YouTube, Unlisted)")
    pdf.set_font("Body", "", 11)
    pdf.block(7, meta.get("video", "REPLACE_WITH_YOUTUBE_URL"))
    pdf.ln(6)
    pdf.set_font("Body", "", 10)
    pdf.block(
        6,
        "Both URLs are required on this title page. The repository must be public, "
        "or private with the module team added as collaborators. The video must open "
        "in a signed-out browser. Word count of the body (sections 1-16) is about "
        "3,000-4,000 words. Appendix A is the responsible-design case and is extra.",
    )
    pdf.add_page()

    in_code = False
    for raw in body_lines:
        line = raw.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if line.startswith("# "):
            continue
        if line.startswith("**Title page**"):
            continue
        if line.startswith("##### "):
            pdf.set_font("Body", "B", 11)
            pdf.ln(2)
            pdf.block(6, wrap_code(line[6:]))
        elif line.startswith("#### "):
            pdf.set_font("Body", "B", 12)
            pdf.ln(3)
            pdf.block(6, wrap_code(line[5:]))
        elif line.startswith("### "):
            pdf.set_font("Body", "B", 12)
            pdf.ln(3)
            pdf.block(6, wrap_code(line[4:]))
        elif line.startswith("## "):
            pdf.set_font("Body", "B", 14)
            pdf.ln(5)
            pdf.block(7, wrap_code(line[3:]))
            pdf.ln(1)
        elif line.startswith("|"):
            pdf.set_font("Body", "", 9)
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            pdf.block(5, "  |  ".join(wrap_code(c) for c in cells))
        elif not line:
            pdf.ln(2)
        else:
            pdf.set_font("Body", "", 10 if in_code else 11)
            pdf.block(5 if in_code else 6, wrap_code(line))
    pdf.output(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    sys.exit(build() or 0)
