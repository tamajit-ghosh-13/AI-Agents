#!/usr/bin/env python3
"""extract_docx.py
Utility script to convert a .docx file to plain text (.txt).
Usage:
    python extract_docx.py --input INPUT.docx --output OUTPUT.txt
The script reads the Word document using the `python-docx` library, extracts all
paragraphs (including line breaks), and writes the concatenated text to the
specified output file.
"""

import argparse
import sys
from pathlib import Path

try:
    import docx
except ImportError as e:
    sys.stderr.write("Error: python-docx is required. Install it via 'pip install python-docx'.\n")
    raise


def docx_to_text(input_path: Path, output_path: Path) -> None:
    """Read a .docx file and write its plain‑text contents to a .txt file.

    Args:
        input_path: Path to the source .docx file.
        output_path: Destination .txt file.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    document = docx.Document(str(input_path))
    text_lines = [para.text for para in document.paragraphs]
    full_text = "\n".join(text_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_text, encoding="utf-8")
    print(f"Converted {input_path} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .docx to .txt")
    parser.add_argument("--input", "-i", required=True, type=Path, help="Path to the .docx file")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Path for the output .txt file")
    args = parser.parse_args()

    try:
        docx_to_text(args.input, args.output)
    except Exception as exc:
        sys.stderr.write(f"Failed to convert: {exc}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
