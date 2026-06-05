# 🇹🇳 tun-camembert-ner

> **Tunisian Named Entity Recognition for French text**, powered by a fine-tuned CamemBERT model.

[![PyPI version](https://badge.fury.io/py/tun-camembert-ner.svg)](https://pypi.org/project/tun-camembert-ner/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-NourBesrour%2Ftun--ner--camembert-yellow)](https://huggingface.co/NourBesrour/tun-ner-camembert)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model](#model)
- [Project Pipeline](#project-pipeline)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

`tun-camembert-ner` is an open-source Python library for **Named Entity Recognition (NER)** in Tunisian French text. It detects and classifies named entities into three categories:

| Entity | Description | Example |
|--------|-------------|---------|
| `PER`  | Person names | `Ahmed Karray`, `Samir Saied` |
| `LOC`  | Cities, regions, countries | `Tunis`, `Sfax`, `Monastir` |
| `ORG`  | Organizations, companies, institutions | `STEG`, `Tunisair`, `BIAT` |

The model is built on top of **CamemBERT** (`camembert-base`), a French BERT model pre-trained on 138GB of French text, fine-tuned on a custom Tunisian French NER dataset.

---

## Features

- ✅ Detects **PER**, **LOC**, and **ORG** entities in French Tunisian text
- ✅ Word-level tokenization — no sub-token splitting
- ✅ Built on CamemBERT — optimized for French
- ✅ Simple and clean Python API
- ✅ CPU and GPU support
- ✅ Lightweight and easy to install

---

## Installation

```bash
pip install tun-camembert-ner
```

Requirements:
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.40+

---

## Quick Start

```python
from tunisian_ner import NER

ner = NER()

result = ner("Ahmed Karray dirige la STEG à Tunis.")
print(result)
```

Output:
```python
[
  {"word": "Ahmed Karray", "entity_group": "PER", "score": 0.999},
  {"word": "STEG",         "entity_group": "ORG", "score": 0.997},
  {"word": "Tunis",        "entity_group": "LOC", "score": 0.999}
]
```

---

## Usage

### Basic usage

```python
from tunisian_ner import NER

ner = NER()

result = ner("Le ministre Samir Saied a visité Sfax hier.")
print(result)
```

### Multiple sentences

```python
sentences = [
    "Ahmed Karray dirige la STEG à Tunis.",
    "Tunisair a annoncé de nouveaux vols vers Paris.",
    "Fatma Mseddi représente Ennahdha à Monastir.",
    "La BIAT a ouvert une nouvelle agence à Sousse.",
]

for sent in sentences:
    print(f"\n📝 {sent}")
    for ent in ner(sent):
        print(f"   → {ent['word']:<25} {ent['entity_group']}  ({ent['score']})")
```

### Filter by entity type

```python
text = "Riadh Bettaieb a signé un accord à Sousse avec la BIAT."
entities = ner(text)

persons   = [e for e in entities if e["entity_group"] == "PER"]
locations = [e for e in entities if e["entity_group"] == "LOC"]
orgs      = [e for e in entities if e["entity_group"] == "ORG"]

print(f"Persons   : {[e['word'] for e in persons]}")
print(f"Locations : {[e['word'] for e in locations]}")
print(f"Orgs      : {[e['word'] for e in orgs]}")
```

### Use a custom model

```python
ner = NER(model="your-username/your-custom-model")
```

---

## Model

The model is hosted on HuggingFace Hub:

🤗 **[NourBesrour/tun-ner-camembert](https://huggingface.co/NourBesrour/tun-ner-camembert)**

| Property | Value |
|----------|-------|
| Base model | `camembert-base` |
| Task | Token Classification (NER) |
| Language | French (Tunisian) |
| Labels | `O`, `B-PER`, `I-PER`, `B-LOC`, `I-LOC`, `B-ORG`, `I-ORG` |
| Training epochs | 10 |
| Max sequence length | 128 tokens |

---

## Project Pipeline

This library was built following a complete NLP pipeline from scratch. Here is the full process:

---

### Step 1 — Data Collection (Web Scraping)

Tunisian French text was collected from multiple online sources using BeautifulSoup and Requests.

**Sources used:**

| Category | Sites |
|----------|-------|
| General news | `lapresse.tn`, `kapitalis.com`, `leaders.com.tn`, `webdo.tn` |
| Economy | `businessnews.com.tn`, `ilboursa.com`, `leaders.com.tn` |
| Politics | `realites.com.tn`, `tap.info.tn`, `presidency.tn` |
| Sport | `sport.tn`, `tunisiesport.net` |
| Tech | `tunisienumerique.com`, `tekiano.com` |
| Wikipedia | `fr.wikipedia.org` (Tunisia articles) |

The scraper collected raw French sentences from article paragraphs, filtered to keep only sentences with at least 3 words.

---

### Step 2 — Data Annotation

Raw sentences were annotated in **BIO (Beginning-Inside-Outside)** format
using **GLiNER** (`urchade/gliner_multi-v2.1`), a zero-shot Named Entity
Recognition model that runs fully locally — no API key needed.

GLiNER predicts entity spans directly from raw text, then the results are
automatically converted to BIO format. A threshold of `0.4` was used to
filter low-confidence predictions.

**How it works:**
- Input: raw French sentence
- GLiNER detects spans for `person`, `location`, `organization`
- Spans are mapped to word-level BIO labels
- Output: `.conll` file ready for training

**BIO format example:**

```
Ahmed       B-PER
Karray      I-PER
dirige      O
la          O
STEG        B-ORG
à           O
Tunis       B-LOC
.           O
```

**Label meaning:**

| Label | Meaning |
|-------|---------|
| `B-PER` | Beginning of a person name |
| `I-PER` | Inside (continuation) of a person name |
| `B-LOC` | Beginning of a location |
| `I-LOC` | Inside a location |
| `B-ORG` | Beginning of an organization |
| `I-ORG` | Inside an organization |
| `O` | Not an entity |

---

### Step 3 — Data Validation & Fixing

The annotated `.conll` file was validated and fixed using custom scripts:

- **`validate_conll.py`** — checks for format errors, unknown labels, and BIO inconsistencies. Generates a visual report showing entity distribution, sentence length histogram, and top entities per type.
- **`fix_bio.py`** — automatically fixes consecutive `B-X B-X` sequences into correct `B-X I-X I-X` tagging.

---

### Step 4 — Dataset Split

The validated dataset was split into 3 subsets:

| Split | Size | Purpose |
|-------|------|---------|
| `train.conll` | 80% | Model learns from this |
| `dev.conll` | 10% | Monitors progress during training |
| `test.conll` | 10% | Final evaluation only |

---

### Step 5 — Model Fine-tuning

The model was fine-tuned on **Google Colab (free T4 GPU)** using the HuggingFace `Trainer` API.

**Base model:** `camembert-base`
- Pre-trained on 138GB of French text
- Robust, stable, and perfectly suited for French NER
- Fully compatible with HuggingFace standard classes

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Batch size | 16 |
| Learning rate | 3e-5 |
| Scheduler | Cosine |
| Max length | 128 tokens |
| Mixed precision | fp16 |
| Best model metric | F1 score |

**Tokenization strategy:**
- Words are pre-split by space before tokenization
- Sub-tokens generated by CamemBERT tokenizer are aligned to their original word
- Only the first sub-token of each word receives the real label
- Other sub-tokens receive label `-100` (ignored in loss computation)

---

### Step 6 — Model Upload to HuggingFace Hub

After training, the model and tokenizer were pushed to HuggingFace Hub using `trainer.push_to_hub()`:

```
https://huggingface.co/NourBesrour/tun-ner-camembert
```

---

### Step 7 — Python Library Packaging

The model was wrapped in a clean Python API and published to PyPI as `tun-camembert-ner`.

**Library structure:**

```
tun-camembert-ner/
├── tunisian_ner/
│   ├── __init__.py       ← from tunisian_ner import NER
│   └── ner.py            ← NER class
├── pyproject.toml        ← package metadata
├── README.md
└── LICENSE
```

**Install:**

```bash
pip install tun-camembert-ner
```

---

## Results

Results on the test set after 10 epochs of fine-tuning:

| Entity | F1 Score |
|--------|----------|
| PER | ~0.99 |
| LOC | ~0.99 |
| ORG | ~0.99 |
| **Overall** | **~0.99** |

---

## Contributing

Contributions are welcome! Here's how to contribute:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/my-feature`
3. Make your changes and commit: `git commit -m "add my feature"`
4. Push to your branch: `git push origin feature/my-feature`
5. Open a Pull Request

**Ideas for contributions:**
- Add more annotated training data
- Support Arabic script input
- Add confidence threshold filtering
- Improve entity boundary detection

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this library in your research, please cite:

```bibtex
@software{besrour2025tunner,
  author    = {Nour Besrour},
  title     = {tun-camembert-ner: Tunisian NER for French text},
  year      = {2025},
  publisher = {PyPI},
  url       = {https://pypi.org/project/tun-camembert-ner/}
}
```

---

## Acknowledgements

- [CamemBERT](https://camembert-model.fr/) — French BERT model by Inria
- [HuggingFace Transformers](https://huggingface.co/transformers/) — model training and inference
- [Google Colab](https://colab.research.google.com/) — free GPU for training
- Tunisian news websites for providing the raw text data