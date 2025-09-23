from gliner import GLiNER

model = GLiNER.from_pretrained(
    "urchade/gliner_medium-v2.1",
)

texts = [
    "pizza and cheese in Austin and California",
    "walmart in washington",
    "uber in Austin",
    "where can I find electronics deals in new york?",
    "home improvement deals in Texas",
]

labels = ["company", "location"]

for text in texts:

    entities = model.predict_entities(
        text,
        labels,
    )

    for entity in entities:
        print(entity["text"], "=>", entity["label"])

model.save_pretrained("models/gliner_medium-v2.1")
