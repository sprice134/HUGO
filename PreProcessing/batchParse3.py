# Created by Stephen Price on March 18th, 2026
# Released under the Apache 2.0 License

import os
from glob import glob
import openai
import pandas as pd
from tqdm import tqdm  # For progress bar

# OpenAI API setup (make sure this matches your working setup!)
with open("../openAiToken.txt", "r") as key_file:
    openai.api_key = key_file.read().strip()

def run_llm(MESSAGES, MODEL='o4-mini', EFFORT='low'):
    """
    Updated for openai-python v1.0+:
    uses namespaced chat.completions.create instead of ChatCompletion.create
    """
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=MESSAGES,
        reasoning_effort=EFFORT
    )
    return resp.choices[0].message.content.strip()

def extract_first_500_chars_from_markdown(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read(500)
    except Exception as e:
        print(f"  Error reading {path}: {e}")
        return ""

def extract_title_from_text(text):
    MESSAGES = [
        {"role": "system", "content": "You are an expert document title extractor."},
        {"role": "user",   "content": f"Extract the title from the following text. Return only the title and nothing else.\n\n{text}"}
    ]
    return run_llm(MESSAGES, MODEL='o4-mini', EFFORT='low')

def load_existing_titles(csv_path):
    """Load existing titles from CSV if it exists."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Ensure it has 'File' and 'Title' columns
        if all(col in df.columns for col in ['File', 'Title']):
            return dict(zip(df['File'], df['Title']))
    return {}

def build_title_dictionary_from_markdowns(directory, existing_titles):
    # 1) Grab _all_ .md files, recursively
    pattern_all = os.path.join(directory, '**', '*.md')
    all_md = glob(pattern_all, recursive=True)
    print(f"DEBUG: glob('{pattern_all}') → {len(all_md)} files found")

    # 2) Filter to just the ones inside an "hybrid_auto" folder
    md_files = [p for p in all_md if os.path.sep + 'hybrid_auto' + os.path.sep in p]
    print(f"DEBUG: filtered to {len(md_files)} files under a 'hybrid_auto/' subdirectory")

    title_dict = {}

    # Use tqdm to show progress on file processing
    for path in tqdm(md_files, desc="Processing Files"):
        # Check if the file is already in the existing titles dictionary
        if path in existing_titles:
            title_dict[path] = existing_titles[path]
            continue

        # If not found, extract the title
        snippet = extract_first_500_chars_from_markdown(path)
        if not snippet:
            title_dict[path] = ""
            continue

        title = extract_title_from_text(snippet)
        title_dict[path] = title

    return title_dict

def save_titles_to_csv(titles_dict, output_file):
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(list(titles_dict.items()), columns=['File', 'Title'])
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved titles to {output_file}")

if __name__ == "__main__":
    markdown_directory = "../SourceArticles/Extractions"
    existing_titles = load_existing_titles("../SourceArticles/Metadata/titles.csv")
    
    # Build title dictionary, skipping files already processed
    titles = build_title_dictionary_from_markdowns(markdown_directory, existing_titles)

    if not titles:
        print("No titles extracted (empty dictionary). Check your paths and glob pattern.")
    else:
        # Save the extracted titles to a new CSV file (titles_new.csv)
        # save_titles_to_csv(titles, "../SourceArticles/Metadata/titles_new.csv")
        save_titles_to_csv(titles, "../SourceArticles/Metadata/titles.csv")

        # Optionally print the titles (or skip this if you only need to save them)
        for path, title in titles.items():
            print(f"File: {path}\n  → Title: {title}\n")
