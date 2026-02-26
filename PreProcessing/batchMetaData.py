#!/usr/bin/env python3
import os
import json
import pandas as pd
import requests
from requests.exceptions import Timeout
from difflib import SequenceMatcher
from tqdm import tqdm

# ─── CONFIG ─────────────────────────────────────────────────────────────────────

INPUT_CSV       = "../SourceArticles/Metadata/titles.csv"                          # must have File, Title, and optionally Link
MASTER_JSON     = "../SourceArticles/Metadata/articlesMetaDataCrossRef_Master.json"
OUTPUT_JSON     = "../SourceArticles/Metadata/articlesMetaDataCrossRef.json"
CROSSREF_URL    = "https://api.crossref.org/works"
SIM_THRESHOLD   = 0.90

# ─── HELPERS ────────────────────────────────────────────────────────────────────

def get_crossref_by_title(title):
    try:
        r = requests.get(CROSSREF_URL, {"query.bibliographic": title, "rows": 1}, timeout=10)
    except Timeout:
        return None
    if r.status_code == 200:
        items = r.json().get("message", {}).get("items", [])
        return items[0] if items else None
    return None

def get_crossref_by_doi(doi):
    clean = doi.strip().lower().removeprefix("doi:")
    try:
        r = requests.get(f"{CROSSREF_URL}/{clean}", timeout=10)
    except Timeout:
        return None
    if r.status_code == 200:
        return r.json().get("message")
    return None

def reduce_metadata(raw):
    title = raw.get("title", ["NA"])[0]
    abstract = raw.get("abstract", "NA")
    date_parts = raw.get("issued", {}).get("date-parts", [])
    published = "-".join(str(x) for x in date_parts[0]) if date_parts else "NA"
    doi = raw.get("DOI", "NA")
    ref_count = raw.get("reference-count", "NA")
    cit_count = raw.get("is-referenced-by-count", "NA")
    publisher = raw.get("publisher", "NA")
    journal = raw.get("container-title", ["NA"])[0]
    issn = ", ".join(raw.get("ISSN", [])) or "NA"
    isbn = ", ".join(raw.get("ISBN", [])) or "NA"
    authors = [
        f"{a.get('given','')} {a.get('family','')}".strip()
        for a in raw.get("author", [])
    ]
    return {
        "Title": title,
        "Abstract": abstract,
        "Published": published,
        "DOI": doi,
        "ReferenceCount": ref_count,
        "Citations": cit_count,
        "Publisher": publisher,
        "Journal": journal,
        "ISSN": issn,
        "ISBN": isbn,
        "Authors": authors
    }

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# ─── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    # load or init cache
    if os.path.exists(MASTER_JSON):
        with open(MASTER_JSON, "r", encoding="utf-8") as mf:
            master_list = json.load(mf)
    else:
        master_list = []
    master_lookup = { e["filename"]: e for e in master_list }

    df = pd.read_csv(INPUT_CSV)
    enriched = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Enriching"):
        fullpath = str(row.get("File", "")).strip()
        title    = str(row.get("Title", "")).strip()
        link     = str(row.get("Link", "")).strip()  # if no Link column, this will be ""
        filename = os.path.basename(fullpath)

        if not title:
            continue

        # reuse if cached
        if filename in master_lookup:
            enriched.append(master_lookup[filename])
            continue

        # fetch metadata
        metadata = None

        # 1) by title
        raw = get_crossref_by_title(title)
        if raw:
            reduced = reduce_metadata(raw)
            if similarity(title.lower(), reduced["Title"].lower()) >= SIM_THRESHOLD:
                metadata = reduced

        # 2) DOI fallback if link is a DOI URL
        if metadata is None and link.lower().startswith("https://doi.org/"):
            doi_str = link.split("doi.org/")[-1]
            raw2 = get_crossref_by_doi(doi_str)
            if raw2:
                metadata = reduce_metadata(raw2)

        if metadata is None:
            metadata = "No metadata found"

        entry = {
            "filename": filename,
            "title": title,
            "link": link,
            "InformationSource": "",
            "DateLabeled": "",
            "InformationInFigures": "",
            "FigureFeatures": "",
            "NumberOfExperiments": "",
            "cold_spray_discussed": "",
            "mechanical_property_present": "",
            "survey_property_present": "",
            "powder_feedstock_recovery": "",
            "alloy_design_discussed": "",
            "brittle_cold_spray_discussed": "",
            "novelty_score": "",
            "keywords": [],
            "extractedText": "",
            "metadata": metadata
        }
        enriched.append(entry)

    # write outputs
    with open(OUTPUT_JSON, "w", encoding="utf-8") as out_f:
        json.dump(enriched, out_f, indent=4)
    with open(MASTER_JSON, "w", encoding="utf-8") as master_f:
        json.dump(enriched, master_f, indent=4)

    print(f"Enriched {len(enriched)} records → {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
