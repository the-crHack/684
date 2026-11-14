"""Verify the report body/reference boundary and render PDFs for visual inspection."""
import argparse
from pathlib import Path

import fitz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--preview-dir", type=Path, default=Path("/tmp/assignment2-pdf-preview"))
    args = parser.parse_args()
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    report = fitz.open(args.directory / "assignment_2_report.pdf")
    reference_pages = [i for i, page in enumerate(report)
                       if any(line.strip() == "References" for line in page.get_text().splitlines())]
    if len(reference_pages) != 1:
        raise ValueError("Cannot identify a unique references heading.")
    body_pages = reference_pages[0]
    if len(report) > 5:
        raise ValueError(f"Entire report exceeds five pages, including references: {len(report)}")
    if body_pages > 5:
        raise ValueError(f"Report body exceeds five pages: {body_pages}")
    if report[body_pages].get_text().splitlines()[0].strip() != "References":
        # Page-number ordering varies, so allow a page number before the heading.
        first_lines = report[body_pages].get_text().splitlines()[:3]
        if "References" not in [line.strip() for line in first_lines]:
            raise ValueError("References do not begin on a separate page.")
    slide = fitz.open(args.directory / "assignment_2_slide.pdf")
    if len(slide) != 1:
        raise ValueError("The presentation must contain exactly one slide.")
    for name, document in [("report", report), ("slide", slide)]:
        for i, page in enumerate(document):
            page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).save(args.preview_dir / f"{name}-{i + 1}.png")
    print(f"Report: {body_pages} body pages, {len(report) - body_pages} reference pages ({len(report)} total); slide: 1 page.")


if __name__ == "__main__":
    main()
