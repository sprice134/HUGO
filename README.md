# HUGO: Document Processing & Extraction

HUGO is a comprehensive framework for batch processing PDF documents, extracting high-fidelity text and figures, and leveraging Large Language Models (LLMs) to extract structured experimental data conforming to defined JSON templates.

---

## Repository Structure

- **[`PreProcessing/`](PreProcessing/)**: Scripts for batch processing PDF documents. Handles parsing, text extraction (via MinerU), figure detection, and metadata retrieval (via CrossRef API).
- **[`Extraction/`](Extraction/)**: Core LLM pipeline. Uses OpenAI models and specific prompts to extract structured data from parsed text.
- **[`PostProcessing/`](PostProcessing/)**: Jupyter Notebooks for data review, standardization, and validation.
- **[`SourceArticles/`](SourceArticles/)**: Primary data directory for raw PDFs and generated outputs.
- **[`HUGO-CS/`](HUGO-CS/)**: Contains the HUGO-CS dataset, ground truth labels, and support files.

---

## Installation Guide

### 1. Environment Setup
```bash
conda create -n hugoEnv python=3.10
conda activate hugoEnv
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install uv
uv pip install -U "mineru[all]"
pip install PyPDF2 pandas openai numpy pymatgen tqdm matplotlib scipy
```

### 3. API Keys
Create an [`openAiToken.txt`](openAiToken.txt) file in the **root** directory containing your OpenAI API key. This is required for both pre-processing and extraction.

---

## Pipeline Execution Overview

### Step 1: Pre-Processing
Place raw PDF files in [`SourceArticles/PDFs/`](SourceArticles/PDFs/). Navigate to [`PreProcessing/`](PreProcessing/) and run the master bash script, which orchestrates the MinerU parsing, figure detection, and metadata retrieval scripts sequentially.
```bash
cd PreProcessing
bash runBatchExtract.sh
```
This populates [`SourceArticles/Extractions/`](SourceArticles/Extractions/) with parsed markdown and [`SourceArticles/Analysis/articlesAnalyzed.json`](SourceArticles/Analysis/articlesAnalyzed.json) with base metadata.

### Step 2: Text Extraction
Navigate to [`Extraction/`](Extraction/):
1. Open [`dataExtraction2.ipynb`](Extraction/dataExtraction2.ipynb) in Jupyter.
2. The notebook reads parsed markdown from [`SourceArticles/Extractions/`](SourceArticles/Extractions/) and metadata from [`SourceArticles/Analysis/articlesAnalyzed.json`](SourceArticles/Analysis/articlesAnalyzed.json).
3. Extraction is guided by prompts in [`HUGO-CS/prompts/`](HUGO-CS/prompts/).
4. Execute cells to generate the structured JSON database.

### Step 3: Post-Processing & Validation
Navigate to [`PostProcessing/`](PostProcessing/) and run the review notebooks:
- [`CategoricalStringProcessing.ipynb`](PostProcessing/CategoricalStringProcessing.ipynb): Standardizes categorical terms and fixes typos.
- [`MaterialCompositionProcessing.ipynb`](PostProcessing/MaterialCompositionProcessing.ipynb): Validates chemical and material compositions.

---

## HUGO-CS Dataset

### Dataset
The HUGO-CS dataset can be found in the [`HUGO-CS/Dataset`](HUGO-CS/Dataset) directory. This dataset contains all meta-data, categorical string mapping and continuous composition processing. Please note that a further processed version of this with standardized unit conversions, as well as a version prior to categorical/continuous string processing can also be found in [`HUGO-CS/Dataset/AlternateVersions`](HUGO-CS/Dataset/AlternateVersions).

### Ground Truth Labels
The ground truth labels for the HUGO-CS dataset can be found in the [`HUGO-CS/GroundTruth/HRM_Flagged`](HUGO-CS/GroundTruth/HRM_Flagged) and [`HUGO-CS/GroundTruth/Held_Out_Val`](HUGO-CS/GroundTruth/Held_Out_Val).

### Support Files
Additional files for HUGO-CS, such as the extraction prompt, the string replacement dictionary, and the schema template can be found in [`HUGO-CS/SupportFiles`](HUGO-CS/SupportFiles).

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.