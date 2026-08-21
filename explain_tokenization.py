
from transformers import AutoTokenizer
from datasets import load_dataset

# Load the same tokenizer we'll use in fine-tuning
print("Loading DistilBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Get some real comments from our dataset
print("Loading a few sample comments from Jigsaw...")
ds = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge")
samples = ds["train"].select(range(3))  # first 3 comments

# Show tokenization for each
for i, sample in enumerate(samples):
    text = sample["comment_text"][:150]  # first 150 chars for readability
    print(f"\n{'─' * 60}")
    print(f"Comment {i+1}: {text}...")

    # Tokenize with our target settings
    encoded = tokenizer(
        text,
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    # Show first 15 tokens (rest is padding)
    print(f"First 15 tokens: {tokens[:15]}")
    print(f"Total tokens (with padding): {len(tokens)}")

    # How much of the input is actual content vs padding?
    real_tokens = sum(encoded["attention_mask"])
    padding = 256 - real_tokens
    print(f"Real tokens: {real_tokens} | Padding: {padding}")

print(f"\n{'─' * 60}")
print("Key insight:")
print("Each comment becomes a fixed-length vector of 256 numbers")
print("regardless of original length. This is what DistilBERT eats.")