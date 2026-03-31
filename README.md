# colrc-ocr-model

A custom OCR (Optical Character Recognition) pipeline for **Coeur d'Alene** documents, combining Tesseract and TrOCR model training with synthetic data generation to recognize historical and specialized typography.

---

## Overview

This project trains and evaluates OCR models on specialized document corpora. It supports two OCR backends — **Tesseract** and **TrOCR (Transformer-based OCR)** — and includes tooling for generating synthetic training data, fine-tuning models, and evaluating character/line-level accuracy.

---

## Project Structure

```
colrc-ocr-model/
├── data_scripts/
│   ├── survey_data.py        # Loads and preprocesses real survey/document data
│   └── synthetic_gen.py      # Generates synthetic training images from fonts
│
├── fonts/                    # Font files used for synthetic data generation
│
├── gen-samples/             # Generated synthetic sample images (output directory)
│
├── model/
│   └── cda.traineddata       # Trained Tesseract model data file
│
├── sources/
│   ├── fonts/
│   │   ├── Charis-Bold.ttf
│   │   ├── Charis-Regular.ttf
│   │   └── DoulosSIL-Regular.ttf
│   └── fonts_config.json     # Font configuration for synthetic generation
│
├── tesseract_scripts/
│   ├── cer.py                # Character Error Rate (CER) evaluation
│   ├── lines.py              # Line-level segmentation and processing
│   ├── new_test.py           # Tesseract evaluation runner (newer)
│
├── trocr_scripts/
│   ├── current.py            # Current TrOCR inference pipeline
│   ├── finetune.py           # Fine-tuning TrOCR on domain-specific data
│   ├── test_finteune.py      # Evaluates fine-tuned TrOCR model
│   ├── test_original.py      # Evaluates baseline (unmodified) TrOCR model
│   └── test.py               # General TrOCR test runner
│
└── test/
    ├── gt/                   # Ground truth transcriptions for evaluation
    ├── images/               # Test document images
    └── pages/                # Full-page test documents
```

---

## Components

### Synthetic Data Generation

Scripts in `data_scripts` generate synthetic training images by rendering text in period-appropriate fonts (Charis, DoulosSIL) defined in `sources/fonts_config.json`. This allows training on typographic styles that match the target documents without requiring large labeled real-world datasets.

### Tesseract Pipeline

The `tesseract_scripts/` directory contains tools for:
- **`cer.py`** — computing Character Error Rate against ground truth
- **`lines.py`** — segmenting documents into line-level units for evaluation
- **`new_test.py`** — running evaluation on test images

The trained model artifact is stored at `model/cda.traineddata` and can be loaded directly by Tesseract via the `--tessdata-dir` flag.

### TrOCR Pipeline

The `trocr_scripts/` directory wraps Microsoft's [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten) transformer model with domain fine-tuning:
- **`finetune.py`** — fine-tunes TrOCR on COLRC-specific image/text pairs
- **`test_original.py`** — evaluates the pre-trained baseline model
- **`test_finteune.py`** — evaluates after fine-tuning
- **`current.py`** — runs inference with the current best model

### Evaluation

Ground truth transcriptions live in `test/gt/` and correspond to images in `test/images/` and `test/pages/`. Evaluation scripts compute CER (Character Error Rate) to compare model outputs against reference text.

---

## Model

The trained Tesseract model (`model/cda.traineddata`) is a custom `.traineddata` file fine-tuned on COLRC document typography. To use it:

```bash
tesseract input.tif output --tessdata-dir ./model -l cda
```

---

## Evaluation Metric

This project uses **Character Error Rate (CER)** as the primary evaluation metric:

```
CER = (Substitutions + Insertions + Deletions) / Total Characters in Ground Truth
```

Lower is better. CER is computed per-line and aggregated across the test set.
