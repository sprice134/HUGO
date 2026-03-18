# Created by Stephen Price on March 18th, 2026
# Released under the Apache 2.0 License

import os
import json
from tqdm import tqdm
from PyPDF2 import PdfReader
import openai

# ─── Configuration ──────────────────────────────────────────────────────────────
CROSSREF_PATH     = "../SourceArticles/Metadata/articlesMetaDataCrossRef.json"
JSON_INPUT_PATH   = "../SourceArticles/Analysis/articlesAnalyzed.json"
JSON_OUTPUT_PATH  = "../SourceArticles/Analysis/articlesAnalyzed.json"
PDF_FOLDER        = "../SourceArticles/PDFs"
MD_FOLDER         = "../SourceArticles/Extractions"
PROMPT_TEMPLATE   = "prompts/parsingPrompt5-26-25.txt"
PAGE_LIMIT        = 40

# ─── OpenAI API Setup ──────────────────────────────────────────────────────────
with open("../openAiToken.txt", "r") as key_file:
    openai.api_key = key_file.read().strip()

# ─── Helper to call LLM ─────────────────────────────────────────────────────────
def run_llm(messages, model='o4-mini', effort='high', temperature=0.0):
    resp = openai.chat.completions.create(
        model=model,
        messages=messages,
        reasoning_effort=effort
    )
    return resp.choices[0].message.content.strip()

# ─── Load system prompt ─────────────────────────────────────────────────────────
with open(PROMPT_TEMPLATE, 'r', encoding='utf-8') as f:
    system_prompt = f.read()

# ─── Load crossref and existing analyzed metadata ────────────────────────────────
print(f"Loading crossref metadata from {CROSSREF_PATH}")
with open(CROSSREF_PATH, 'r', encoding='utf-8') as f:
    crossref = json.load(f)
print(f"Found {len(crossref)} entries in crossref metadata")

if os.path.exists(JSON_INPUT_PATH):
    print(f"Loading existing analyzed metadata from {JSON_INPUT_PATH}")
    with open(JSON_INPUT_PATH, 'r', encoding='utf-8') as f:
        analyzed = json.load(f)
    print(f"Found {len(analyzed)} entries in analyzed metadata")
else:
    analyzed = []
    print("No existing analyzed metadata found; starting from crossref only")

# Build lookup for quick merge
analyzed_dict = {art.get("filename", ""): art for art in analyzed}

# Merge: for each crossref entry, use analyzed version if present, else crossref block
articles = []
for entry in crossref:
    fname = entry.get("filename", "")
    if fname in analyzed_dict:
        articles.append(analyzed_dict[fname])
    else:
        articles.append(entry)
print(f"Prepared {len(articles)} total articles to analyze ({len(articles) - len(analyzed)} new)")

# ─── Process each article ──────────────────────────────────────────────────────
for article in tqdm(articles, desc="Analyzing articles"):
    md_filename = article.get("filename", "")
    base        = os.path.splitext(md_filename)[0]
    pdf_filename = base + ".pdf"

    pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
    md_path  = os.path.join(MD_FOLDER, base, "hybrid_auto", f"{base}.md")

    # Initialize missing fields
    for key in (
        "extractedText",
        "cold_spray_discussed",
        "mechanical_property_present",
        "survey_property_present",
        "powder_feedstock_recovery",
        "alloy_design_discussed",
        "brittle_cold_spray_discussed",
        "keywords",
        "novelty_score"
    ):
        if key not in article:
            article[key] = "" if key != "novelty_score" else None

    # Skip if already processed
    if isinstance(article.get("novelty_score"), int):
        continue

    # 1) PDF existence check
    if not os.path.exists(pdf_path):
        article["extractedText"] = "Pruned - Article not found"
        continue

    # 2) Markdown existence check
    if not os.path.exists(md_path):
        article["extractedText"] = "Pruned - Markdown not found"
        continue

    # 3) Page count check
    try:
        reader = PdfReader(pdf_path)
        if len(reader.pages) > PAGE_LIMIT:
            article["extractedText"] = f"Pruned - Length Exceeds {PAGE_LIMIT} pages"
            continue
    except Exception as e:
        article["extractedText"] = f"Pruned - Error reading PDF: {e}"
        continue

    # 4) Load markdown content
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            extracted_text = f.read()
    except Exception as e:
        article["extractedText"] = f"Pruned - Error reading markdown: {e}"
        continue

    # 5) Build and send LLM prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": extracted_text}
    ]
    try:
        result = run_llm(messages)
        data = json.loads(result)
    except Exception as e:
        article["extractedText"] = f"Pruned - LLM Error: {e}"
        continue

    # 6) Populate fields from LLM response
    article["cold_spray_discussed"]         = data.get("cold_spray_discussed", "")
    article["mechanical_property_present"]  = data.get("mechanical_property_present", "")
    article["survey_property_present"]      = data.get("survey_property_present", "")
    article["powder_feedstock_recovery"]    = data.get("powder_feedstock_recovery", "")
    article["alloy_design_discussed"]       = data.get("alloy_design_discussed", "")
    article["brittle_cold_spray_discussed"] = data.get("brittle_cold_spray_discussed", "")
    article["keywords"]                     = data.get("keywords", [])
    article["novelty_score"]                = data.get("novelty_score", None)

    # Save incrementally
    os.makedirs(os.path.dirname(JSON_OUTPUT_PATH), exist_ok=True)
    with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as out_f:
        json.dump(articles, out_f, indent=4)

# Final save
os.makedirs(os.path.dirname(JSON_OUTPUT_PATH), exist_ok=True)
with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as out_f:
    json.dump(articles, out_f, indent=4)
print(f"Analysis complete. Results written to {JSON_OUTPUT_PATH}")
