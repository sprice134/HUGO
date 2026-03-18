# Created by Stephen Price on March 18th, 2026
# Released under the Apache 2.0 License

import os
import glob
import subprocess
import random
from PyPDF2 import PdfReader

# === CONFIGURATION ===
PDF_DIR    = "../SourceArticles/PDFs"
OUTPUT_DIR = "../SourceArticles/Extractions"
PAGE_LIMIT = 650
LANG       = "en"

# === HELPERS ===

def get_page_count(path):
    """Return number of pages in the PDF at `path`."""
    with open(path, "rb") as f:
        return len(PdfReader(f).pages)

def already_processed_set():
    """
    Return a set of basenames for which OUTPUT_DIR/basename/.../*.md exists.
    """
    processed = set()
    if not os.path.isdir(OUTPUT_DIR):
        return processed

    for subdir in os.listdir(OUTPUT_DIR):
        full_sub = os.path.join(OUTPUT_DIR, subdir)
        if not os.path.isdir(full_sub):
            continue
        basename = subdir
        pattern = os.path.join(full_sub, "**", f"{basename}.md")
        if glob.glob(pattern, recursive=True):
            processed.add(basename)
    return processed

def clean_intermediates(basename):
    """Remove all intermediate files under OUTPUT_DIR/basename."""
    out_subdir = os.path.join(OUTPUT_DIR, basename)
    patterns = [
        f"{basename}_content_list.json",
        f"{basename}_content_list_v2.json",
        f"{basename}_middle.json",
        f"{basename}_model.json",
        f"{basename}_origin.pdf",
        f"{basename}_layout.pdf",
        f"{basename}_spans.pdf"
    ]
    for pattern in patterns:
        for path in glob.glob(os.path.join(out_subdir, "**", pattern), recursive=True):
            try:
                os.remove(path)
                print(f"  ✔ Removed intermediate: {path}")
            except OSError as e:
                print(f"  ✖ Failed to remove {path}: {e}")

def process_pdf(pdf_path):
    """
    Run magic-pdf on `pdf_path`, print stdout/stderr, then clean up.
    Returns True on zero exit code, False otherwise.
    """
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"\n→ Processing {basename} ...")
    cmd = [
        "mineru",
        "-p", pdf_path,
        "-o", OUTPUT_DIR,
        "--lang", LANG
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("  → mineru stdout:")
    print(result.stdout or "    <no stdout>")
    print("  → mineru stderr:")
    print(result.stderr or "    <no stderr>")

    if result.returncode != 0:
        print(f"  ✖ Error processing {basename}: exit code {result.returncode}")
        return False

    print(f"  ✅ Completed {basename}")
    clean_intermediates(basename)
    return True

# === MAIN LOOP ===

def main():
    # persistent in-memory processed set so we don't re-pick successes or skips
    processed = already_processed_set()

    while True:
        # scan for all PDFs
        pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
        if not pdfs:
            print(f"No PDFs found in {PDF_DIR}")
            return

        # build list of candidates not yet in processed
        candidates = []
        for pdf in pdfs:
            basename = os.path.splitext(os.path.basename(pdf))[0]
            if basename not in processed:
                candidates.append((basename, pdf))

        if not candidates:
            print("All found PDFs have been processed or skipped.")
            return

        # pick one at random
        basename, pdf_to_do = random.choice(candidates)

        # check page count
        pages = get_page_count(pdf_to_do)
        if pages > PAGE_LIMIT:
            print(f"— Skipping {basename}: {pages} pages > {PAGE_LIMIT}.")
            processed.add(basename)
            continue

        # process it
        success = process_pdf(pdf_to_do)
        # mark done/failed so we don't pick again
        processed.add(basename)

if __name__ == "__main__":
    main()
