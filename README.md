# HUGO Document Processing & Extraction Repository

This repository consolidates the pipeline for document pre-processing, large language model (LLM) based text extraction, and post-processing/validation. 

## Repository Structure

- **`PreProcessing/`**: Contains scripts to batch process PDF documents. This phase handles PDF parsing, text extraction (using MinerU), figure detection, and metadata retrieval (via CrossRef API).
- **`Extraction/`**: Contains the core LLM pipeline. It takes the parsed text and uses OpenAI models (via `dataExtraction2.ipynb`) along with specific prompts to extract structured experimental data conforming to defined JSON templates. 
- **`PostProcessing/`**: Contains Jupyter Notebooks used for data review and standardization. This step maps variables to standard categories and processes complex material compositions.
- **`SourceArticles/`**: The primary data directory. Place raw PDFs in `SourceArticles/PDFs`. Outputs from PreProcessing are saved to `SourceArticles/Analysis`, `SourceArticles/Extractions`, and `SourceArticles/Metadata`.

---

## Installation Guide

The entire pipeline can be run using a unified conda environment.

### 1. Environment Setup

It is highly recommended to use Miniconda or Anaconda. Create a new environment named `hugoEnv` running Python 3.10:

```bash
conda create -n hugoEnv python=3.10
conda activate hugoEnv
```

### 2. Install Dependencies

Install the core components, including MinerU for PDF parsing, and standard data science libraries:

```bash
pip install --upgrade pip
pip install uv
uv pip install -U "mineru[all]"
pip install PyPDF2 pandas openai numpy pymatgen tqdm matplotlib scipy
```

### 3. API Keys

Create an `openAiToken.txt` file in the **root** of the `HUGO` repository containing your OpenAI API key. This is required by both the pre-processing parsing scripts and the main extraction notebooks.

---

## Pipeline Execution Overview

### Step 1: Pre-Processing

Place your raw PDF files in the `SourceArticles/PDFs/` directory. Navigate to `PreProcessing/` and run the master bash script, which orchestrates the MinerU parsing, figure detection, and metadata retrieval scripts sequentially.

```bash
cd PreProcessing
bash runBatchExtract.sh
```

This populates `SourceArticles/Extractions/` with parsed markdown and `SourceArticles/Analysis/articlesAnalyzed.json` with the base metadata.

### Step 2: Text Extraction

Navigate to `Extraction/`. 

1. Open `dataExtraction2.ipynb` in Jupyter Notebook or JupyterLab.
2. The notebook will automatically read the parsed markdown from `SourceArticles/Extractions/` and the metadata from `SourceArticles/Analysis/articlesAnalyzed.json`.
3. The notebook uses prompts located in `../HUGO-CS/prompts/` to guide the LLM extraction. 
4. Execute the cells to process the articles and generate the final structured JSON database.

### Step 3: Post-Processing & Validation

Navigate to `PostProcessing/`.
Run the review notebooks to clean the extracted JSON results:
- `CategoricalStringProcessing.ipynb`: Standardizes categorical strings (e.g., fixing typos and mapping to standardized terms).
- `MaterialCompositionProcessing.ipynb`: Checks and standardizes chemical and material compositions extracted by the model.
