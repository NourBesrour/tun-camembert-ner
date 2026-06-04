# tunisian-ner

Tunisian Named Entity Recognition for French text using CamemBERT.

Detects 3 entity types:
- **PER** — person names
- **LOC** — cities and regions
- **ORG** — organizations

## Installation

pip install tunisian-ner

## Usage

from tunisian_ner import NER

ner = NER()
results = ner("Ahmed Karray dirige la STEG à Tunis.")
print(results)
# → [{"word": "Ahmed Karray", "entity_group": "PER", "score": 0.97},
#    {"word": "STEG", "entity_group": "ORG", "score": 0.95},
#    {"word": "Tunis", "entity_group": "LOC", "score": 0.98}]

## Model

Fine-tuned CamemBERT on Tunisian French text.
HuggingFace: NourBesrour/tun-ner-camembert