#!/usr/bin/env python3
import os
import json
from datetime import date
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm

# ─── Configuration ──────────────────────────────────────────────────────────────
JSON_PATH       = "../SourceArticles/Analysis/articlesAnalyzed.json"
MD_FOLDER       = "../SourceArticles/Extractions"                       # where your hybrid_auto/*.md lives
PROMPT_TEMPLATE = "prompts/parsingPrompt_FigureDetection_6-13-25.txt"
MAX_WORKERS     = 16         # adjust as needed
TOKEN_LIMIT     = 1_950_000  # stop issuing new API calls after this many total tokens

# derive today's label
today_str = date.today().isoformat()

# ─── OpenAI client setup ─────────────────────────────────────────────────────────
with open("../openAiToken.txt", "r") as key_file:
    api_key = key_file.read().strip()
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI(api_key=api_key)

def run_llm(messages, model="o3", effort="medium"):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        # reasoning_effort=effort
    )
    content    = resp.choices[0].message.content.strip()
    tokens_used = resp.usage.total_tokens
    return content, tokens_used

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
    os.replace(tmp, path)

def process_one(idx, article, system_prompt):
    """
    Worker thread: read the markdown for this article, call the LLM
    to get data_category, return the result dict.
    """
    # only proceed if mechanical_property_present is YES and not yet labeled
    if article.get("mechanical_property_present", "").upper() != "YES":
        return None
    if "data_category" in article:
        return None

    # locate and load markdown
    fname = article.get("filename", "")
    base  = os.path.splitext(fname)[0]
    md_path = os.path.join(MD_FOLDER, base, "hybrid_auto", f"{base}.md")
    if not os.path.exists(md_path):
        # nothing to do
        return None

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read().strip()
    if not md_text:
        return None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"```{md_text}```"}
    ]
    raw, used_tokens = run_llm(messages)

    # strip any code fences and parse
    cleaned = (raw
        .replace("```json\n", "")
        .replace("\n```", "")
    )
    try:
        parsed = json.loads(cleaned)
        data_cat = parsed.get("data_category")
    except Exception:
        # malformed JSON; skip writing but account for tokens
        return {"idx": idx, "tokens": used_tokens}

    # build result dict: always include data_category; if it's 2, also include the extras
    result = {
        "idx": idx,
        "data_category": data_cat,
        "tokens": used_tokens
    }
    if data_cat == 2:
        result["num_experiments"] = parsed.get("num_experiments")
        result["figure_metrics"] = parsed.get("figure_metrics")
    return result

def main():
    # 1) Load metadata
    articles = load_json(JSON_PATH)

    # 2) Load system prompt
    with open(PROMPT_TEMPLATE, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # 3) Build job list
    jobs = [
        (i, art)
        for i, art in enumerate(articles)
        if art.get("mechanical_property_present", "").upper() == "YES"
           and "data_category" not in art
    ]

    token_count = 0
    job_iter = iter(jobs)

    # 4) Process with thread pool + token budget
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        # initial fill
        while len(futures) < MAX_WORKERS and token_count < TOKEN_LIMIT:
            try:
                idx, art = next(job_iter)
            except StopIteration:
                break
            fut = pool.submit(process_one, idx, art, system_prompt)
            futures[fut] = idx

        with tqdm(total=len(jobs), desc="Assigning data_category", unit="article") as pbar:
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    idx = futures.pop(fut)
                    result = fut.result()
                    if result:
                        art = articles[result["idx"]]
                        art["data_category"] = result["data_category"]
                        if result.get("data_category") == 2:
                            art["num_expected_experiments_in_figures"] = result.get("num_experiments")
                            art["properties_reported_in_figures"] = result.get("figure_metrics")
                        token_count += result["tokens"]
                        save_json(articles, JSON_PATH)
                    pbar.update(1)

                # submit more if under token limit
                while len(futures) < MAX_WORKERS and token_count < TOKEN_LIMIT:
                    try:
                        idx, art = next(job_iter)
                    except StopIteration:
                        break
                    fut = pool.submit(process_one, idx, art, system_prompt)
                    futures[fut] = idx

    # 5) Final save
    save_json(articles, JSON_PATH)
    print(f"Data-category assignment complete. Results written to {JSON_PATH}")

if __name__ == "__main__":
    main()
