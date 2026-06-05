from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch


class NER:
    def __init__(self, model="NourBesrour/tun-ner-camembert"):
        print("Loading model...")
        self.device    = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model     = AutoModelForTokenClassification.from_pretrained(model)
        self.model.eval()
        if self.device == 0:
            self.model = self.model.cuda()
        self.id2label  = self.model.config.id2label
        print("✅ Model ready!")

    def __call__(self, text: str):
        # split by space — one prediction per word
        words  = text.split()
        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        if self.device == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predictions = torch.argmax(logits, dim=-1)[0].tolist()
        word_ids    = self.tokenizer(
            words, is_split_into_words=True
        ).word_ids()

        # one label per word (first sub-token only)
        word_labels = {}
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id not in word_labels:
                word_labels[word_id] = self.id2label[predictions[idx]]

        # build entities
        entities, current = [], None
        for word_id, word in enumerate(words):
            label = word_labels.get(word_id, "O")

            if label.startswith("B-"):
                if current: entities.append(current)
                current = {
                    "word"        : word,
                    "entity_group": label[2:],
                    "score"       : 1.0,
                }
            elif label.startswith("I-") and current and current["entity_group"] == label[2:]:
                current["word"] += " " + word
            else:
                if current: entities.append(current)
                current = None

        if current:
            entities.append(current)

        return entities