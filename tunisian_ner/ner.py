from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class NER:
    def __init__(self, model="NourBesrour/tun-ner-camembert"):
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.pipe = pipeline(
            "ner",
            model=model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=-1    # CPU; change to 0 for GPU
        )
        print("✅ Model ready!")

    def __call__(self, text: str):
        results = self.pipe(text)
        return [
            {
                "word"        : ent["word"],
                "entity_group": ent["entity_group"],
                "score"       : round(ent["score"], 4),
            }
            for ent in results
        ]