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
│   ├── survey_data.py        # Linguistic analysis on synthetic data
│   └── synthetic_gen.py      # Generates synthetic training images
│
├── gen-samples/             # Generated synthetic sample images (output directory)
│
├── model/
│   └── rcd.traineddata       # Trained Tesseract model data file (Short for Reichard, to differentiate from CdA from other project)
│
├── sources/
│   ├── fonts/
│   │   ├── Charis-Bold.ttf
│   │   ├── Charis-Regular.ttf
│   │   └── DoulosSIL-Regular.ttf
│   └── fonts_config.json     # Font configuration for synthetic generation
│
├── tesseract_scripts/
│   ├── new_test.py           # Tesseract evaluation runner (newer)
│   ├── per_line_cer.py                # Character Error Rate (CER) evaluation on every line in a folder
│   ├── total_cer.py              # Overall CER evaluation for model evaluation folder
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
    ├── pages/               # Full-page test documents
    ├── pred/               # Prediction outputs on eval data from model
    └── pred-base/                # Prection outputs on eval data from base (Latin) model
```

---

## Components

### Synthetic Data Generation

Scripts in `data_scripts` generate synthetic training images by rendering text in period-appropriate fonts (Charis, DoulosSIL) defined in `sources/fonts_config.json`. This allows training on typographic styles that match the target documents without requiring large labeled real-world datasets.

### Tesseract Pipeline

The `tesseract_scripts/` directory contains tools for:
- **`page_test.py`** — Performs OCR and extracts text from inputed file


To test a specific model to compare models create a folder - pred and run:

```bash
for img in test/images/*.png; do
  base=$(basename "$img" .png)
  tesseract "$img" "test/pred/$base" \
    -l my_lang \
    --psm 6 \
    --oem 1
done
```

This will perform OCR on your evaluation images on a specific model and output it, then you can run the following test models: 

- **`per_line_cer.py`** — gathers the CER on each line of test data vs model's output of the test data
- **`total_cer.py`** — calculates the overall CER of model output

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

The trained Tesseract model (`model/cda.traineddata`) is a custom `.traineddata` file fine-tuned on COLRC document typography. To use on a single image in CdA and in console:


### CLI Outputs
Extract OCR'd text to a file:
```bash
tesseract input.png output_file_name --tessdata-dir ./model -l rcd
```

Print in console:
```bash
tesseract input.png stdout --tessdata-dir ./model -l rcd
``` 


### Python Script Multi-page Output

OR to export text or print an entire book/pages run

`tesseract_scripts/page_test`



Note though, with testing a whole page/pdf is difficult. The source texts are sometimes uneven lines of CdA and English. While Tesseract does have a co-language paramater rcd+eng, it doesn't work extraordinarily well. But where it does sense rcd it does so relatively accurately.

---

## Next Steps

Find a way to OCR these specific typed Reichard documents:
Options:
- split the lines into different images and run the relative OCR language model

- Optimize the scanned pdfs for the lines to be straighter and overall less noisy to improve smoother extraction


Overall next steps for documents already in CdA:

- Create a website interface for community members to upload their photograhs
