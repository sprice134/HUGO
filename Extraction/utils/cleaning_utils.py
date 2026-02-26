# cleaning_utils.py

import os
import re
import json
import copy
from collections import Counter
from difflib import SequenceMatcher
import numpy as np


# =============================
# Article clipping
# =============================
def article_clipping(input_path: str, output_path: str, clipping_thresh: int, print_removed=False) -> None:
    """
    Load a JSON list of article dicts, drop any with filename 'Article_<id>.md'
    where <id> > clipping_thresh, and save to output_path. Prints a summary.
    """
    # === Load data ===
    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # === Filter: drop Article_X.md where X > clipping_thresh ===
    pat = re.compile(r"^Article_(\d+)\.md$")

    removed = []
    kept = []
    for d in docs:
        fn = d.get("filename", "")
        m = pat.match(fn)
        if m and int(m.group(1)) > int(clipping_thresh):
            removed.append(fn)
        else:
            kept.append(d)

    # === Save output ===
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    # === Summary ===
    print(f"Filtering Summary (Clipping all values over {clipping_thresh})")
    print(f"  Total entries in input:  {len(docs)}")
    print(f"  Entries kept:            {len(kept)}")
    print(f"  Entries removed:         {len(removed)}")
    if removed and print_removed:
        print("\nRemoved filenames:")
        for fn in removed:
            print(f"  - {fn}")

    print(f"Filtered JSON written to: {output_path}\n")


# =============================
# Duplicate Removal
# =============================

def clean_duplicates_by_config(input_path: str, config_path: str, output_path: str) -> None:
    """
    Loads a main article list and a configuration JSON.
    Removes any article whose filename appears in the 'Strip' list 
    of any entry in the configuration file.
    """
    # === 1. Load Data ===
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # === 2. Build Denylist ===
    # We only care about the 'Strip' lists for removal. 
    # The 'Keep' list is useful for your record-keeping but technically 
    # doesn't change the logic (anything not Stripped is kept by default).
    files_to_remove = set()
    
    for doi, instructions in config.items():
        strip_list = instructions.get("Strip", [])
        for fname in strip_list:
            files_to_remove.add(fname)

    # === 3. Filter ===
    kept_docs = []
    removed_docs = []

    for d in docs:
        fn = d.get("filename", "")
        if fn in files_to_remove:
            removed_docs.append(fn)
        else:
            kept_docs.append(d)

    # === 4. Save Output ===
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kept_docs, f, indent=2, ensure_ascii=False)

    # === 5. Summary ===
    print("Duplicate Cleaning Summary")
    print("-" * 30)
    print(f"  Configuration loaded from: {os.path.basename(config_path)}")
    print(f"  Total entries in input:    {len(docs)}")
    print(f"  Entries removed:           {len(removed_docs)}")
    print(f"  Entries kept:              {len(kept_docs)}")
    print("-" * 30)
    
    # Optional: Validate that we actually found the files we wanted to strip
    # (Check if the config asked to strip files that weren't in the input)
    input_filenames = {d.get("filename") for d in docs}
    missed_strips = files_to_remove - input_filenames
    if missed_strips:
        print(f"  Warning: {len(missed_strips)} files listed in 'Strip' were not found in the input.")
    
    print(f"\nCleaned JSON written to: {output_path}")


# =============================
# Ground-truth replace
# =============================
def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _contains_unchecked(obj):
    """
    Recursively search any nested dict/list for a value equal to "unchecked" (case-insensitive).
    """
    if isinstance(obj, dict):
        return any(_contains_unchecked(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_unchecked(v) for v in obj)
    if isinstance(obj, str):
        return obj.strip().lower() == "unchecked"
    return False


def _gather_valid_ground_truth(gt_dir):
    """
    Return a dict mapping article_id (string, e.g. "157") to its JSON
    for all files Article_<id>.json that do NOT contain "Unchecked".
    """
    pattern = re.compile(r"^Article_(\d+)\.json$")
    valid = {}
    for fn in os.listdir(gt_dir):
        m = pattern.match(fn)
        if not m:
            continue
        article_id = m.group(1)
        path = os.path.join(gt_dir, fn)
        data = _load_json(path)
        if not _contains_unchecked(data):
            valid[article_id] = data
    return valid


def _apply_ground_truth(original_docs, gt_map, HELD_OUT=False):
    """
    For each doc in original_docs, if its filename matches Article_<id>.md
    and we have a ground-truth for <id>, update only its Experiments (and title).
    Everything else remains untouched. If HELD_OUT is True, set InformationSource to "Ground Truth (Validation)".
    
    Parameters:
    ----------
    original_docs : list
        List of documents to process.
    gt_map : dict
        Ground truth mapping (article_id -> ground-truth data).
    HELD_OUT : bool, optional
        If True, will set InformationSource to "Ground Truth (Validation)".
        Default is False.
    """
    filename_pattern = re.compile(r"Article_(\d+)\.md$")
    # ---- Added logging counters ----
    total_added = 0
    total_removed = 0
    total_modified = 0
    # --------------------------------
    for doc in original_docs:
        fn = os.path.basename(doc.get("filename", ""))
        m = filename_pattern.match(fn)
        if not m:
            # filename doesn't look like Article_<id>.md → skip
            continue

        article_id = m.group(1)
        gt = gt_map.get(article_id)
        if not gt:
            # we have no ground truth for this id → skip
            continue

        # Make sure extractedText is dict
        ext = doc.get("extractedText")
        if not isinstance(ext, dict):
            ext = {}

        # ---- Logging computation (before mutation) ----
        old_exps = ext.get("Experiments", [])
        old_len = len(old_exps) if isinstance(old_exps, list) else 0
        new_exps = gt.get("Experiments", [])
        new_len = len(new_exps) if isinstance(new_exps, list) else 0

        shared = min(old_len, new_len)
        added = max(0, new_len - old_len)
        removed = max(0, old_len - new_len)
        modified = shared

        total_added += added
        total_removed += removed
        total_modified += modified
        # ------------------------------------------------

        # 1) Replace the entire extractedText block (keep only Experiments and optional title)
        ext = {"Experiments": gt.get("Experiments", [])}
        # 2) (Optional) replace the title inside extractedText
        if "title" in gt:
            ext["title"] = gt["title"]

        doc["extractedText"] = ext

        # 3) Record where this data came from
        if HELD_OUT:
            doc["InformationSource"] = "Ground Truth (Validation)"
        else:
            doc["InformationSource"] = "Ground Truth"
        
        # 4) How many experiments we just wrote in
        doc["NumberOfExperiments"] = len(ext["Experiments"])
        # 5) When that ground truth was labeled
        doc["DateLabeled"] = gt.get("DateLabeled", doc.get("DateLabeled"))

    # ---- Summary ----
    print(f"  TOTAL — modified: {total_modified}, added: {total_added}, removed: {total_removed}")
    return original_docs


def _count_experiments_in_gt(gt_map):
    """
    Count total number of experiments across all valid ground-truth articles.
    """
    total = 0
    for gt in gt_map.values():
        exps = gt.get("Experiments", [])
        if isinstance(exps, list):
            total += len(exps)
    return total


def gt_replace(input_path: str, output_path: str, gt_dir: str, HELD_OUT=False) -> None:
    """
    Merge valid ground-truth experiments into the extracted dataset.
    Calls _apply_ground_truth to update documents.
    
    Parameters
    ----------
    input_path : str
        Path to the extracted dataset JSON to update.
    output_path : str
        Path to write the updated JSON.
    gt_dir : str
        Directory containing Article_<id>.json ground-truth files.
    HELD_OUT : bool, optional
        If True, will set InformationSource to "Ground Truth (Validation)".
    """
    # 1. load all valid ground-truth JSONs
    print('Ground Truth Replacement Summary')
    gt_map = _gather_valid_ground_truth(gt_dir)
    print(f"  Found {len(gt_map)} valid ground-truth articles.")
    total_gt_experiments = _count_experiments_in_gt(gt_map)
    print(f"  Ground-truth contains {total_gt_experiments} experiments in total.")

    # 2. load original extraction
    all_docs = _load_json(input_path)
    print(f"  Original JSON has {len(all_docs)} documents.")

    # 3. apply updates
    updated_docs = _apply_ground_truth(all_docs, gt_map, HELD_OUT)

    # 4. save merged output
    _save_json(updated_docs, output_path)
    print(f"Wrote updated JSON to {output_path}\n")


# =============================
# Remapping keys to template
# =============================
def remapping_keys(input_path: str, output_path: str, template_path: str) -> None:
    """
    Remap keys in the dataset to the closest keys present in the template schema.
    Writes a remapped JSON and prints a summary.
    """
    # === Load template schema ===
    with open(template_path, 'r', encoding='utf-8') as f:
        template = json.load(f)
    template_exp = template["Experiments"][0]

    # Build sets of valid keys for each section
    valid_keys = {
        "preSprayedProperties": set(template_exp.get("preSprayedProperties", {}).keys()),
        "experimentalProperties": set(template_exp.get("experimentalProperties", {}).keys()),
        "resultsValues": set(template_exp.get("resultsValues", {}).keys()),
    }

    # Flatten all valid keys
    all_valid_keys = set().union(*valid_keys.values())

    # === Load dataset ===
    with open(input_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    # === Identify and count keys ===
    counter = Counter()
    total_keys = 0

    for doc in docs:
        ext = doc.get("extractedText", {})
        if not isinstance(ext, dict):
            continue
        exps = ext.get("Experiments", [])
        if not isinstance(exps, list):
            continue
        for exp in exps:
            if not isinstance(exp, dict):
                continue
            for section in ("preSprayedProperties", "experimentalProperties", "resultsValues"):
                props = exp.get(section, {})
                if not isinstance(props, dict):
                    continue
                for key in list(props.keys()):
                    total_keys += 1
                    if key not in valid_keys[section]:
                        counter[key] += 1

    # === Build recommendation map ===
    recommendation_map = {}
    for key in counter:
        best = None
        best_score = 0.0
        for valid in all_valid_keys:
            score = SequenceMatcher(None, key.lower(), valid.lower()).ratio()
            if score > best_score:
                best_score = score
                best = valid
        # only map if similarity > 0, else leave unmapped
        if best_score > 0:
            recommendation_map[key] = best

    # === Remap keys in a fresh copy of the data ===
    remapped_docs = copy.deepcopy(docs)

    for doc in remapped_docs:
        ext = doc.get("extractedText", {})
        if not isinstance(ext, dict):
            continue
        exps = ext.get("Experiments", [])
        if not isinstance(exps, list):
            continue
        for exp in exps:
            if not isinstance(exp, dict):
                continue
            for section in ("preSprayedProperties", "experimentalProperties", "resultsValues"):
                props = exp.get(section, {})
                if not isinstance(props, dict):
                    continue
                for key in list(props.keys()):
                    if key in recommendation_map:
                        # rename to recommended key
                        new_key = recommendation_map[key]
                        props[new_key] = props.pop(key)

    # === Save remapped JSON ===
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(remapped_docs, f, indent=2, ensure_ascii=False)

    # === Summary Prints ===
    print("Re-mapping keys to expected template (using closest match via SequenceMatcher)")
    total_incorrect = sum(counter.values())
    high_sim_instances = sum(
        freq for key, freq in counter.items()
        if recommendation_map.get(key)
        and SequenceMatcher(None, key.lower(), recommendation_map[key].lower()).ratio() > 0.90
    )
    total_correct = total_keys - total_incorrect

    print("Summary:")
    print(f"  Total keys processed:              {total_keys}")
    print(f"  Total correct instances:           {total_correct}")
    print(f"  Total incorrect instances:         {total_incorrect}")
    print(f"  Instances remapped (sim>0):        {sum(counter.values())}")
    print(f"  Instances with similarity >0.90:   {high_sim_instances}")
    print(f"Remapped JSON written to: {output_path}\n")


def swap_misassigned_features(input_path: str, output_path: str, template_path: str) -> None:
    """
    Re-align properties across sections by moving keys into the correct section
    according to the template. Writes a swapped JSON and prints a move summary.
    """
    # Load template
    with open(template_path, 'r', encoding='utf-8') as f:
        template = json.load(f)
    template_exp = template["Experiments"][0]

    # Build sets of valid keys per section
    sections = ["preSprayedProperties", "experimentalProperties", "resultsValues"]
    valid_keys = {sect: set(template_exp.get(sect, {}).keys()) for sect in sections}

    # Load dataset
    remapped_docs = _load_json(input_path)

    # Re-align with full source→target moves
    total_checked = 0
    move_log = []  # tuples of (key, from_section, to_section)

    for doc in remapped_docs:
        ext = doc.get("extractedText", {})
        if not isinstance(ext, dict):
            continue
        exps = ext.get("Experiments", [])
        if not isinstance(exps, list):
            continue

        for exp in exps:
            if not isinstance(exp, dict):
                continue

            # ensure sections exist
            for sect in sections:
                exp.setdefault(sect, {})

            # scan each section as source
            for src in sections:
                props = exp[src]
                for key in list(props.keys()):
                    total_checked += 1
                    # find the correct section for this key
                    for tgt in sections:
                        if key in valid_keys[tgt]:
                            if tgt != src:
                                print(f"Moving key '{key}' from '{src}' → '{tgt}'")
                                move_log.append((key, src, tgt))
                                exp[tgt][key] = exp[src].pop(key)
                            break

    # Summary
    print("Re-alignment Summary (swap misassigned features to correct sections)")
    print(f"  Total properties checked: {total_checked}")
    print(f"  Total properties moved:   {len(move_log)}")

    from collections import Counter
    cnt = Counter((src, tgt) for _, src, tgt in move_log)
    print("  Breakdown of moves:")
    for (src, tgt), freq in cnt.items():
        print(f"    {freq:4d}   {src:20s} → {tgt}")

    # Save
    _save_json(remapped_docs, output_path)
    print(f"Remapped & re-aligned JSON written to: {output_path}\n")

def delete_empty_experiments(input_path: str, output_path: str) -> None:
    """
    Delete any experiment whose 'resultsValues' exists and all its values are empty strings.
    Writes a pruned JSON and prints a summary.
    """
    docs = _load_json(input_path)
    removed_experiments = 0

    for doc in docs:
        ext = doc.get("extractedText")
        if not isinstance(ext, dict):
            continue
        exps = ext.get("Experiments")
        if not isinstance(exps, list):
            continue

        filtered = []
        for exp in exps:
            if not isinstance(exp, dict):
                continue
            rv = exp.get("resultsValues")
            if isinstance(rv, dict) and rv and all(v == "" for v in rv.values()):
                removed_experiments += 1
                print("Deleting experiment (all empty resultsValues):", exp)
            else:
                filtered.append(exp)

        ext["Experiments"] = filtered

    _save_json(docs, output_path)
    print("Delete-Empty-Experiments Summary (remove experiments with empty resultsValues)")
    print(f"  Total experiment blocks removed: {removed_experiments}")
    print(f"Pruned JSON written to: {output_path}\n")



def fill_missing_values(input_path: str, output_path: str, template_path: str) -> None:
    """
    Keep only valid keys (matching the template) with scalar types, then fill any missing
    keys per section with '' (or 'False' for *_Binary). Prints a detailed summary.
    """
    # Load template
    with open(template_path, 'r', encoding='utf-8') as f:
        template = json.load(f)
    template_exp = template["Experiments"][0]

    valid_keys = {
        section: set(template_exp.get(section, {}).keys())
        for section in ("preSprayedProperties", "experimentalProperties", "resultsValues")
    }

    # Load dataset
    docs = _load_json(input_path)

    # Counters for reporting
    removed_count     = 0
    filled_count      = 0
    modified_docs     = 0
    binary_set_count  = 0

    # Track which files had removals, and what keys were removed
    removed_fields = {}  # filename -> list of removed keys

    def get_doc_filename(doc, idx):
        return doc.get("filename") or doc.get("file_name") or f"doc_{idx}"

    # Clean each document
    for idx, doc in enumerate(docs):
        doc_modified = False
        fname = get_doc_filename(doc, idx)

        ext = doc.get("extractedText", {})
        if not isinstance(ext, dict):
            continue
        exps = ext.get("Experiments", [])
        if not isinstance(exps, list):
            continue

        for exp in exps:
            if not isinstance(exp, dict):
                continue

            # For each section: prune invalid & non‐(string/number/bool), then fill missing
            for section, keys in valid_keys.items():
                props = exp.get(section, {})
                if not isinstance(props, dict):
                    props = {}
                new_props = {}

                # 1) keep only exact‐match keys with str, int, float or bool values
                for k, v in props.items():
                    if k in keys and isinstance(v, (str, bool, int, float)):
                        new_props[k] = v
                    else:
                        removed_count += 1
                        doc_modified = True
                        removed_fields.setdefault(fname, []).append(f"{section}.{k}")

                # 2) fill in any missing keys
                for k in keys:
                    if k not in new_props:
                        if k.endswith("_Binary"):
                            new_props[k] = "False"
                            binary_set_count += 1
                        else:
                            new_props[k] = ""
                        filled_count += 1
                        doc_modified = True
                    else:
                        # If key exists and ends with _Binary and is empty string, set to "False"
                        if k.endswith("_Binary") and new_props[k] == "":
                            new_props[k] = "False"
                            binary_set_count += 1
                            doc_modified = True

                exp[section] = new_props

        if doc_modified:
            modified_docs += 1

    # Save
    _save_json(docs, output_path)

    # Summary
    print("Fill-Missing-Values Summary (retain valid keys; fill missing; set *_Binary='False')")
    print(f"  Total documents processed:           {len(docs)}")
    print(f"  Documents modified:                  {modified_docs}")
    print(f"  Values removed (invalid type/key):   {removed_count}")
    print(f"  Values filled (missing keys):        {filled_count}")
    print(f"  Binaries set to False:               {binary_set_count}")

    print("\nDeleted fields by file:")
    for fname, keys in removed_fields.items():
        print(f"\nFile: {fname}")
        for key in keys:
            print(f"  - {key}")

    print(f"\nCleaned JSON written to: {output_path}")


def sort_keys(input_path: str, output_path: str, template_path: str) -> None:
    import json
    from collections import OrderedDict
    from copy import deepcopy

    # --- Load template with guaranteed order ---
    with open(template_path, "r", encoding="utf-8") as f:
        template = json.load(f, object_pairs_hook=OrderedDict)

    # Grab the experiment template (first example)
    exp_tmpl = (template.get("Experiments") or [{}])[0]
    # Build explicit ordered key-lists for each section
    pre_keys  = list(exp_tmpl.get("preSprayedProperties", {}).keys())
    exp_keys  = list(exp_tmpl.get("experimentalProperties", {}).keys())
    res_keys  = list(exp_tmpl.get("resultsValues", {}).keys())
    # Notes is just a scalar in your template; we don't need a key list for inside Notes

    def order_section(sec_dict, ordered_keys):
        """
        Return a new OrderedDict where:
          1) Keys present in 'ordered_keys' appear first, in that order (only if they exist in sec_dict)
          2) Any remaining (extra) keys are appended alphabetically
        """
        if not isinstance(sec_dict, dict):
            return sec_dict
        out = OrderedDict()
        # 1) template-ordered keys
        for k in ordered_keys:
            if k in sec_dict:
                out[k] = sec_dict[k]
        # 2) extras alphabetically
        extras = sorted(k for k in sec_dict.keys() if k not in out)
        for k in extras:
            out[k] = sec_dict[k]
        return out

    def sort_one_experiment(exp):
        if not isinstance(exp, dict):
            return exp
        out = OrderedDict()
        # Ensure the 4 sections appear in the same order as the template (and sorted inside)
        if "preSprayedProperties" in exp:
            out["preSprayedProperties"] = order_section(exp["preSprayedProperties"], pre_keys)
        if "experimentalProperties" in exp:
            out["experimentalProperties"] = order_section(exp["experimentalProperties"], exp_keys)
        if "resultsValues" in exp:
            out["resultsValues"] = order_section(exp["resultsValues"], res_keys)
        # 'Notes' (and any other extra sections) appended after the three main ones, alphabetically by section name
        extra_sections = sorted(
            k for k in exp.keys()
            if k not in ("preSprayedProperties", "experimentalProperties", "resultsValues")
        )
        for sec in extra_sections:
            out[sec] = exp[sec]
        return out

    def sort_experiments_in_doc(doc):
        if not isinstance(doc, dict):
            return doc

        # Case A: top-level Experiments
        if isinstance(doc.get("Experiments"), list):
            doc["Experiments"] = [sort_one_experiment(e) for e in doc["Experiments"]]

        # Case B: nested under extractedText
        ext = doc.get("extractedText")
        if isinstance(ext, dict) and isinstance(ext.get("Experiments"), list):
            ext["Experiments"] = [sort_one_experiment(e) for e in ext["Experiments"]]

        return doc

    # --- Load data (order not required here) ---
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Optional: quick diagnostics to catch template mismatches ---
    def peek_mismatches(data, limit=3):
        """
        Print a few examples where a section has keys not in the template.
        This helps diagnose if you're accidentally pointing at a slightly different template.
        """
        shown = 0
        for doc in data:
            ext = doc.get("extractedText", {})
            exps = []
            if isinstance(ext, dict) and isinstance(ext.get("Experiments"), list):
                exps = ext["Experiments"]
            elif isinstance(doc.get("Experiments"), list):
                exps = doc["Experiments"]

            for e in exps:
                for sec, key_list in (
                    ("preSprayedProperties", pre_keys),
                    ("experimentalProperties", exp_keys),
                    ("resultsValues", res_keys),
                ):
                    sec_dict = e.get(sec, {})
                    if isinstance(sec_dict, dict):
                        extras = sorted(set(sec_dict.keys()) - set(key_list))
                        missing = sorted(set(key_list) - set(sec_dict.keys()))
                        if extras or missing:
                            print(f"— Mismatch in {doc.get('filename', '<unknown>')} / {sec}")
                            if extras:  print("   Extras not in template:", extras[:10], ("... +%d more" % (len(extras)-10) if len(extras)>10 else ""))
                            if missing: print("   Missing from data:", missing[:10], ("... +%d more" % (len(missing)-10) if len(missing)>10 else ""))
                            shown += 1
                            if shown >= limit:
                                return

    # --- Sort all docs ---
    sorted_data = [sort_experiments_in_doc(deepcopy(doc)) for doc in data]

    # --- Save preserving the order we built ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=2, ensure_ascii=False)

    print(f"Sorted JSON written to: {output_path}")


def value_grouping(input_path: str, output_path: str, mapping_path: str) -> None:
    # ─── LIBRARIES ──────────────────────────────────────────────────────────────
    import json
    from collections import Counter

    # ─── LOAD MAPPING DICTS ──────────────────────────────────────────────────────
    with open(mapping_path, 'r') as f:
        mapping_src = json.load(f)

    # ─── helper to modularize string replacement for map keys ───────────────────
    def lower_keys(d: dict) -> dict:
        return { (k.lower() if isinstance(k, str) else k): v for k, v in d.items() }

    # existing maps
    material_map    = lower_keys(mapping_src.get('materialNamingConventions', {}))
    element_map     = lower_keys(mapping_src.get('elementMapping', {}))
    gas_map         = lower_keys(mapping_src.get('gasMapping', {}))
    system_map      = lower_keys(mapping_src.get('systemNameMapping', {}))
    powder_map      = lower_keys(mapping_src.get('powderManufacturing', {}))
    orientation_map = lower_keys(mapping_src.get('tensileOrientation', {}))
    standard_map    = lower_keys(mapping_src.get('tensileStandardMapping', {}))
    # new composition map
    chemcomp_map    = lower_keys(mapping_src.get('chemicalCompositionMapping', {}))

    # new atomic‐percent composition map
    chemcomp_atomic_map = lower_keys(mapping_src.get('chemicalCompositionMappingAtomic', {}))

    # ─── ATOMIC → WEIGHT‐PERCENT HELPER ───────────────────────────────────────────
    from pymatgen.core.composition import Composition
    from pymatgen.core.periodic_table import Element

    def atomic_map_to_wtjson(full_atomic_map):
        prefix = "Majority_Powder_"
        suffix = "_Percentage"

        # 1) pull out just the at.% values into a dict element→float
        at_pct = {}
        for key, val in full_atomic_map.items():
            if not key.endswith(suffix) or val == "":
                continue
            el = key[len(prefix):-len(suffix)]
            try:
                at_pct[el] = float(val)
            except ValueError:
                continue

        # 2) build a Composition (it normalizes mole fractions automatically)
        comp = Composition({el: pct/100.0 for el, pct in at_pct.items()})

        # 3) get mole‐amounts for each element
        el_amounts = comp.get_el_amt_dict()  # e.g. {"Co": 0.25, "Cr": 0.25, ...}

        # 4) compute mass contributions using Element.atomic_mass
        mass_contrib = {}
        for sym, amt in el_amounts.items():
            atomic_wt = Element(sym).atomic_mass
            mass_contrib[sym] = amt * atomic_wt

        total_mass = sum(mass_contrib.values())

        # 5) compute weight‐fractions and rebuild the full map
        wt_map = {}
        for key, val in full_atomic_map.items():
            if not key.endswith(suffix):
                wt_map[key] = val
            else:
                el = key[len(prefix):-len(suffix)]
                if el in mass_contrib and total_mass > 0:
                    wt_pct = (mass_contrib[el] / total_mass) * 100.0
                    wt_map[key] = f"{wt_pct:.4f}"
                else:
                    wt_map[key] = val

        return json.dumps(wt_map, ensure_ascii=False)

    # ─── NORMALIZATION HELPERS ───────────────────────────────────────────────────
    def normalize(name, lookup, default="[V] Other"):
        """
        Case-insensitive lookup with:
        - blank / None / "[V] Not Reported" -> "[V] Not Reported"
        - mapped values -> mapped value
        - everything else unmapped -> default (typically "[V] Other")
        """
        if name is None:
            return "[V] Not Reported"

        if isinstance(name, str):
            s = name.strip()
            if s == "" or s.lower() == "[v] not reported":
                return "[V] Not Reported"
            key = s.lower()
            return lookup.get(key, default)

        # non-string: coerce to string and treat similarly
        s = str(name).strip()
        if s == "" or s.lower() == "[v] not reported":
            return "[V] Not Reported"
        return lookup.get(s.lower(), default)


    def normalize_and_stringify(val, lookup, default=None):
        """
        Normalize val via lookup. If the replacement is a dict or list,
        JSON-dump it back to a string.
        """
        if not isinstance(val, str):
            return val
        key = val.strip().lower()
        if key not in lookup:
            return default if default is not None else val.strip()
        replacement = lookup[key]
        if isinstance(replacement, (dict, list)):
            return json.dumps(replacement, ensure_ascii=False)
        return replacement

    # ─── UTILITY TO PRINT VALUE COUNTS (OPTIONAL) ────────────────────────────────
    def print_unique_value_counts(experiments, section, feature):
        counter = Counter()
        key2val = {}
        for exp in experiments:
            if not isinstance(exp, dict): continue
            props = exp.get(section, {})
            if not isinstance(props, dict): continue
            val = props.get(feature)
            if val is None: continue
            key = json.dumps(val, sort_keys=True) if isinstance(val, (dict, list)) else str(val)
            counter[key] += 1
            key2val[key] = val
        print(f"\nFeature '{feature}' in '{section}':")
        for key, cnt in counter.most_common():
            print(f"  {key2val[key]!r}: {cnt}")
        print(f"Total unique: {len(counter)}")

    # ─── PROCESS & NORMALIZE ───────────────────────────────────────────────────
    with open(input_path, 'r') as f:
        articles = json.load(f)

    all_experiments = []
    replacement_counts = Counter()

    for art in articles:
        ext = art.get('extractedText')
        if isinstance(ext, str):
            try:
                ext = json.loads(ext)
            except json.JSONDecodeError:
                continue
        if not isinstance(ext, dict):
            continue

        exps = ext.get('Experiments', [])
        for exp in exps:
            if not isinstance(exp, dict):
                continue
            all_experiments.append(exp)

            pre  = exp.setdefault('preSprayedProperties', {})
            prop = exp.setdefault('experimentalProperties', {})

            # 1) Materials
            for key in (
                'Majority_Powder_Material_Name',
                'Secondary_Powder_Material_Name',
                'Tertiary_Powder_Material_Name'
            ):
                if key in pre:
                    old = pre[key]
                    new = normalize(pre[key], material_map, pre[key])
                    if new != old:
                        replacement_counts[key] += 1
                    pre[key] = new

            # 1b) Chemical Composition normalization via chemcomp_map
            for key in (
                'Majority_Powder_Chemical_Composition',
                'Secondary_Powder_Chemical_Composition',
                'Tertiary_Powder_Chemical_Composition'
            ):
                if key in pre:
                    old = pre[key]
                    new = normalize_and_stringify(pre[key],
                                                  chemcomp_map,
                                                  pre[key])
                    if new != old:
                        replacement_counts[key] += 1
                    pre[key] = new

            # ─── fallback atomic‐percent → weight‐percent ──────────────────────────
            for key in (
                'Majority_Powder_Chemical_Composition',
                'Secondary_Powder_Chemical_Composition',
                'Tertiary_Powder_Chemical_Composition'
            ):
                raw = pre.get(key, "")
                lower = raw.strip().lower()
                if lower not in chemcomp_map and lower in chemcomp_atomic_map:
                    atomic_data = chemcomp_atomic_map[lower]
                    if isinstance(atomic_data, str):
                        try:
                            atomic_data = json.loads(atomic_data)
                        except json.JSONDecodeError:
                            atomic_data = {}
                    try:
                        old = pre.get(key, "")
                        pre[key] = atomic_map_to_wtjson(atomic_data)
                        if pre[key] != old:
                            replacement_counts[key] += 1
                    except Exception as e:
                        print(f"Warning: couldn’t convert atomic→weight for '{raw}': {e}")
            
            # 1B) Primary Element (Majority / Secondary / Tertiary)
            for key in (
                "Majority_Powder_Primary_Element",
                "Secondary_Powder_Primary_Element",
                "Tertiary_Powder_Primary_Element",
            ):
                if key in pre:
                    old = pre.get(key, "")
                    new = normalize(old, element_map, old)
                    if new != old:
                        replacement_counts[key] += 1
                    pre[key] = new
                    
            # 2) Process Gas
            gas = prop.get('Process_Gas_Type', "")
            new_gas = normalize(gas, gas_map, "[V] Other")
            if new_gas != gas:
                replacement_counts['Process_Gas_Type'] += 1
            prop['Process_Gas_Type'] = new_gas

            # 3) Spraying System
            sysm = prop.get('Spraying_System_Model', "")
            new_sysm = normalize(sysm, system_map, "[V] Other")
            if new_sysm != sysm:
                replacement_counts['Spraying_System_Model'] += 1
            prop['Spraying_System_Model'] = new_sysm

            # 4) Powder Production
            pcm = pre.get('Powder_Production_Method', "")
            new_pcm = normalize(pcm, powder_map, "[V] Other")
            if new_pcm != pcm:
                replacement_counts['Powder_Production_Method'] += 1
            pre['Powder_Production_Method'] = new_pcm

            # 5) Tensile Orientation
            ori = prop.get('Tensile_Test_Orientation', "")
            new_ori = normalize(ori, orientation_map, "[V] Other")
            if new_ori != ori:
                replacement_counts['Tensile_Test_Orientation'] += 1
            prop['Tensile_Test_Orientation'] = new_ori

            # 6) Standards
            stan = prop.get('Standard_for_Tensile_Testing', "")
            new_stan = normalize(stan, standard_map, "[V] Other")
            if new_stan != stan:
                replacement_counts['Standard_for_Tensile_Testing'] += 1
            prop['Standard_for_Tensile_Testing'] = new_stan
            

        art['extractedText'] = ext

    # ─── SAVE NORMALIZED JSON ────────────────────────────────────────────────────
    with open(output_path, 'w') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    # ─── PRINT REPLACEMENT COUNTS ────────────────────────────────────────────────
    print("\nReplacement counts per key:")
    for k, v in replacement_counts.items():
        print(f"  {k}: {v}")

    print(f"Normalized JSON written to: {output_path}")



def impute_material_compositions(input_path: str, output_path: str, mapping_path: str) -> None:
    #!/usr/bin/env python3
    import json

    # ─── LOAD MAPPING + IMPUTATION TABLE ──────────────────────────────────────────
    with open(mapping_path, 'r') as f:
        mapping_src = json.load(f)

    # imputation table: lower-case keys for case-insensitive matching
    raw_imp = mapping_src.get('chemicalCompositionImputation', {})
    imp_map = {k.strip().lower(): v for k, v in raw_imp.items()}

    # ─── LOAD DATA ────────────────────────────────────────────────────────────────
    with open(input_path, 'r') as f:
        articles = json.load(f)

    # ─── RUN IMPUTATION ───────────────────────────────────────────────────────────
    blank_vals = {"", "[v] not reported"}
    for art in articles:
        ext = art.get('extractedText')
        if isinstance(ext, str):
            try:
                ext = json.loads(ext)
            except json.JSONDecodeError:
                continue
        if not isinstance(ext, dict):
            continue

        exps = ext.get('Experiments', [])
        for exp in exps:
            if not isinstance(exp, dict):
                continue

            pre = exp.setdefault('preSprayedProperties', {})

            # loop over majority, secondary, tertiary
            for role in ('Majority', 'Secondary', 'Tertiary'):
                comp_key = f'{role}_Powder_Chemical_Composition'
                mat_key  = f'{role}_Powder_Material_Name'

                comp_val = pre.get(comp_key)
                mat_name = (pre.get(mat_key) or "").strip()
                # treat empty, None, or "[V] Not Reported" (case-insensitive) as blank
                if mat_name and (comp_val is None or comp_val.strip().lower() in blank_vals):
                    lookup_key = mat_name.lower()
                    if lookup_key in imp_map:
                        pre[comp_key] = json.dumps(
                            imp_map[lookup_key],
                            ensure_ascii=False
                        )

        art['extractedText'] = ext

    # ─── SAVE OUTPUT ──────────────────────────────────────────────────────────────
    with open(output_path, 'w') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"Imputed JSON written to: {output_path}")


def blend_and_mix(input_path: str, output_path: str, mapping_path: str) -> None:
    #!/usr/bin/env python3
    import json
    from collections import defaultdict

    def is_false(val):
        return val is False or (isinstance(val, str) and val.strip().lower() == 'false')

    def is_blank_or_not_reported(val):
        if val is None:
            return True
        if isinstance(val, str):
            s = val.strip()
            if s == "":
                return True
            if s.lower() == "[v] not reported":
                return True
        return False

    def is_json_string(s):
        if not isinstance(s, str):
            return False
        try:
            parsed = json.loads(s)
            return isinstance(parsed, (dict, list))
        except:
            return False

    def parse_composition(comp_str):
        """
        comp_str: JSON string of {"Elem": "value", ...}
        returns: dict Elem -> float
        """
        data = json.loads(comp_str)
        out = {}
        for k, v in data.items():
            try:
                out[k] = float(v) if v not in (None, "", "[V] Not Reported") else 0.0
            except:
                out[k] = 0.0
        return out

    def mix_compositions(comps, ratios):
        """
        comps: list of dicts (Elem->float)
        ratios: list of floats summing to 1
        returns dict Elem->float (weighted sum)
        """
        mixed = defaultdict(float)
        for comp_dict, ratio in zip(comps, ratios):
            for elem, pct in comp_dict.items():
                mixed[elem] += pct * ratio
        return mixed

    # load regex lookup
    with open(mapping_path, 'r') as f:
        mapping_src = json.load(f)
    raw_blend = mapping_src.get('BlendRatioStandardization', {})
    blend_map = {k.strip().lower(): v for k, v in raw_blend.items()}

    # load data
    with open(input_path, 'r') as f:
        articles = json.load(f)

    for art in articles:
        raw_ext = art.get('extractedText')
        was_str = isinstance(raw_ext, str)
        if was_str:
            try:
                ext = json.loads(raw_ext)
            except:
                continue
        else:
            ext = raw_ext
        if not isinstance(ext, dict):
            continue

        for exp in ext.get('Experiments', []):
            if not isinstance(exp, dict):
                continue
            pre = exp.get('preSprayedProperties')
            if not isinstance(pre, dict):
                continue

            # ---------- NEW RULE ----------
            # If Multiple_Powders_Binary == False and both secondary/tertiary comps are blank
            # or "[V] Not Reported", set Powder_Blend_Ratio_Standardized = "100".
            mpb = pre.get('Multiple_Powders_Binary')
            sec = pre.get('Secondary_Powder_Chemical_Composition')
            ter = pre.get('Tertiary_Powder_Chemical_Composition')
            if is_false(mpb) and is_blank_or_not_reported(sec) and is_blank_or_not_reported(ter):
                pre['Powder_Blend_Ratio_Standardized'] = "100"

            # 1) impute blend atomic composition if missing (original behavior)
            comp = pre.get('Majority_Powder_Chemical_Composition')
            if is_false(mpb) and is_json_string(comp) and not pre.get('Powder_Blend_Atomic_Composition'):
                pre['Powder_Blend_Atomic_Composition'] = comp

            # 2) standardize blend ratio (DO NOT overwrite if already set by the new rule)
            if not str(pre.get('Powder_Blend_Ratio_Standardized', "")).strip():
                raw_ratio = pre.get('Powder_Blend_Ratio', "")
                if isinstance(raw_ratio, str) and raw_ratio.strip().lower() in blend_map:
                    std = blend_map[raw_ratio.strip().lower()]
                    pre['Powder_Blend_Ratio_Standardized'] = std

            # 3) mix compositions if needed
            existing_val = pre.get('Powder_Blend_Atomic_Composition', "")
            existing = existing_val.strip() if isinstance(existing_val, str) else str(existing_val).strip()
            std = pre.get('Powder_Blend_Ratio_Standardized')
            if (not existing) and isinstance(std, str):
                # parse ratio fractions
                parts = std.split(':')
                try:
                    ratios = [float(p) for p in parts]
                    total = sum(ratios)
                    ratios = [r/total for r in ratios]
                except:
                    continue

                # collect component JSONs
                comp_fields = [
                    'Majority_Powder_Chemical_Composition',
                    'Secondary_Powder_Chemical_Composition',
                    'Tertiary_Powder_Chemical_Composition'
                ]
                comps = []
                valid = True
                for idx, ratio in enumerate(ratios):
                    if idx >= len(comp_fields):
                        valid = False
                        break
                    field = comp_fields[idx]
                    s = pre.get(field)
                    if not is_json_string(s):
                        valid = False
                        break
                    comps.append(parse_composition(s))
                if not valid:
                    continue

                # mix & write back
                mixed = mix_compositions(comps, ratios)
                mixed_str = {k: f"{v:.4f}".rstrip('0').rstrip('.') for k, v in mixed.items()}
                pre['Powder_Blend_Atomic_Composition'] = json.dumps(mixed_str, ensure_ascii=False)

        # write back ext
        art['extractedText'] = json.dumps(ext, ensure_ascii=False) if was_str else ext

    # save
    with open(output_path, 'w') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"Blended & mixed data written to {output_path}")



# =============================
# Group treatments (robust; maps empty string to Not Reported)
# =============================
import json, os, re, unicodedata
from collections import Counter

def _norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def _safe_get_experiments(extracted):
    if isinstance(extracted, str):
        try:
            extracted = json.loads(extracted)
        except Exception:
            return []
    if isinstance(extracted, dict):
        exps = extracted.get("Experiments", [])
        return exps if isinstance(exps, list) else []
    if isinstance(extracted, list):
        out = []
        for item in extracted:
            if isinstance(item, dict):
                exps = item.get("Experiments", [])
                if isinstance(exps, list):
                    out.extend(exps)
        return out
    return []

def group_treatments(input_path: str, output_path: str, mapping_path: str) -> None:
    """
    Adds experimentalProperties.Treatment_Categorical based on
    experimentalProperties.Deposit_Post_Treatment_Description using a mapping.

    - Leaves the original Deposit_Post_Treatment_Description untouched.
    - Matching is case-insensitive and Unicode/whitespace-normalized.
    - Maps empty/None descriptions if mapping includes "" (e.g., "[V] Not Reported").
    """
    # 1) Load mapping
    with open(mapping_path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    treatment_map = raw_map.get("treatmentMapping", raw_map)
    norm_map = { _norm_text(k): v for k, v in treatment_map.items() if isinstance(k, str) or k is None }

    # 2) Load dataset
    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # 3) Iterate and map
    total_exps = 0
    with_field = 0
    mapped = 0
    with_field_unmapped = 0
    missing = 0
    unmapped_counter = Counter()

    target_key = "Deposit_Post_Treatment_Description"

    for doc in docs:
        exps = _safe_get_experiments(doc.get("extractedText", {}))
        for exp in exps:
            if not isinstance(exp, dict):
                continue
            total_exps += 1

            props = exp.get("experimentalProperties")
            if not isinstance(props, dict):
                missing += 1
                continue

            # Count "with field" if the key exists (even if "", None, etc.)
            if target_key in props:
                with_field += 1
                raw_val = props.get(target_key, "")
                key = _norm_text(raw_val)
                cat = norm_map.get(key)
                if cat is not None:
                    props["Treatment_Categorical"] = cat
                    mapped += 1
                else:
                    with_field_unmapped += 1
                    # keep original value in the unmapped diagnostics
                    unmapped_counter[str(raw_val)] += 1
                exp["experimentalProperties"] = props
            else:
                # truly missing key
                missing += 1

    # 4) Save output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    # 5) Summary
    print("Treatment grouping summary")
    print(f"  Total experiments:                {total_exps}")
    print(f"  ├─ with description field:        {with_field}")
    print(f"  │   ├─ mapped to category:        {mapped}")
    print(f"  │   └─ present but unmapped:      {with_field_unmapped}")
    print(f"  └─ missing description field:     {missing}")
    print(f"Updated JSON written to: {output_path}\n")

    if unmapped_counter:
        print("Top unmapped Deposit_Post_Treatment_Description values:")
        for val, cnt in unmapped_counter.most_common(20):
            print(f"  {cnt} × {val!r}")
        print()


import json
import re
import numpy as np

def _classify_powder_morphology(input):
    # Handle cases like "Not Reported" or empty strings
    if isinstance(input, str):
        input = input.strip()  # Strip any leading/trailing whitespace
        if input in ["", "[V] Not Reported", "None", "NaN"]:
            return 'Unknown'  # Return 'Unknown' for these cases, no error print

    # Process float cases (e.g., NaN)
    if isinstance(input, float):
        if np.isnan(input):
            return 'Unknown'  # Return 'Unknown' for NaN, no error print

    if isinstance(input, str):
        # Split if multiple values
        if '; ' in input or '. ' in input:
            input_ls = re.split(r'; |\. ', input)
            valid_inputs = []
            throwaway_inputs = []
            for input_piece in input_ls:
                if 'Al ' in input_piece:
                    valid_inputs.append(input_piece)
                else:
                    throwaway_inputs.append(input_piece)

            if len(valid_inputs) == 0:
                pass
            elif len(valid_inputs) > 1:
                return f'Error -- multiple valid inputs {valid_inputs}'
            else:
                return _classify_powder_morphology(valid_inputs[0])

        # Regular classification logic
        if input in ['NaN', 'None', '[V] Not Reported']:
            return '[V] Unknown'
        elif 'dendrit' in input.lower() or 'electrolytic' in input.lower():
            return '[V] Dendritic'
        elif 'lamella' in input.lower() or 'lath' in input.lower():
            return '[V] Lamellar'
        elif 'strip' in input.lower() or 'worm' in input.lower():
            return '[V] Strip'
        elif 'disk' in input.lower() or 'plate' in input.lower() or 'sheet' in input.lower() or 'flake' in input.lower():
            return '[V] Plate'
        elif 'angular' in input.lower() or 'crystalline' in input.lower() or 'block' in input.lower() or 'polygon' in input.lower():
            return '[V] Angular'
        elif 'spher' in input.lower() and ('satellite' in input.lower() or 'rough' in input.lower()) or 'globular protrusions' in input.lower() or 'adhered' in input.lower():
            return '[V] Spheroid w/ satellites'
        elif 'irreg' in input.lower() and 'spher' in input.lower():
            return '[V] Irregular spheroid'
        elif 'oblate' in input.lower() or 'elongated' in input.lower() or 'oblong' in input.lower() or 'globular' in input.lower():
            return '[V] Oblong'
        elif 'spong' in input.lower() or 'porous' in input.lower():
            return '[V] Sponge-like'
        elif 'agglom' in input.lower():
            return '[V] Agglomerated'
        elif 'irreg' in input.lower() or 'various shape' in input.lower() or 'rugged' in input.lower() or 'protuberant' in input.lower() or 'rough' in input.lower() or 'crushed' in input.lower() or 'convoluted' in input.lower() or 'decorated' in input.lower():
            return '[V] Irregular'
        elif 'spher' in input.lower() \
                or 'shpercial' in input.lower() \
                or 'sperical' in input.lower() \
                or 'excellent morphology' in input.lower() \
                or 'round' in input.lower() \
                or 'subglobular' in input.lower() \
                or 'circular' in input.lower() \
                or 'equiaxed' in input.lower() \
                or 'smooth surface' in input.lower() \
                or 'spray-dried' in input.lower():
            return '[V] Spheroid'
        else:
            print(f"Error: Invalid morphology input '{input}'")
            return '[V] Unknown'  # Print error for invalid values but return 'Unknown'
    else:
        print(f"Error: Unexpected input type {type(input)} for value '{input}'")
        return '[V] Unknown'  # Print error for unexpected types
    

def classify_and_save(input_path, output_path):
    """
    Load the JSON data, apply the powder morphology classification, and save the result.
    
    Parameters
    ----------
    input_path : str
        Path to the input JSON file containing the dataset.
    output_path : str
        Path to save the updated JSON file.
    """
    # Load the JSON dataset
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize counters for summary
    total_processed = 0
    total_skipped = 0
    total_errors = 0

    # Apply the morphology classification to each article
    for article in data:
        extracted_text = article.get("extractedText", {})
        if isinstance(extracted_text, dict):
            experiments = extracted_text.get("Experiments", [])
            # Apply classification directly here
            for exp in experiments:
                if isinstance(exp, dict):
                    props = exp.get("preSprayedProperties", {})
                    if isinstance(props, dict):
                        powder_morphology = props.get("Powder_Morphology", "")
                        # Apply the classification function
                        if isinstance(powder_morphology, str):
                            powder_morphology = powder_morphology.strip()
                            if powder_morphology in ["", "[V] Not Reported", "None", "NaN"]:
                                classified_morphology = 'Unknown'
                                total_skipped += 1  # Count as skipped due to missing data
                            elif '; ' in powder_morphology or '. ' in powder_morphology:
                                input_ls = re.split(r'; |\. ', powder_morphology)
                                valid_inputs = []
                                for input_piece in input_ls:
                                    if 'Al ' in input_piece:
                                        valid_inputs.append(input_piece)
                                classified_morphology = _classify_powder_morphology(valid_inputs[0]) if valid_inputs else 'Unknown'
                                total_processed += 1
                            else:
                                classified_morphology = _classify_powder_morphology(powder_morphology)
                                total_processed += 1
                        else:
                            classified_morphology = 'Unknown'
                            total_skipped += 1  # Count as skipped due to missing or invalid data
                        # Update the 'Powder_Morphology' field
                        props["Powder_Morphology"] = classified_morphology
            # Update the experiments in the article
            extracted_text["Experiments"] = experiments

    # Save the updated dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Print summary metrics
    print(f"Updated JSON with classified Powder_Morphology saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Total entries processed:  {total_processed}")
    print(f"  Total entries skipped (missing or invalid data): {total_skipped}")
    print(f"  Total entries skipped (errors): {total_errors}")
