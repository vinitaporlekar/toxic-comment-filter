# toxic-comment-filter

Real-time toxic comment classifier. Fine-tunes DistilBERT on the Jigsaw Toxic Comments dataset to detect toxicity, insults, threats, and identity hate.

## Status

🚧 In development — see [PROGRESS.md](./PROGRESS.md) for daily updates.

## Planned architecture

- Fine-tuned `distilbert-base-uncased` on Jigsaw Toxic Comments
- FastAPI serving layer with async inference
- Streamlit demo UI
- Deployed on HuggingFace Spaces

## Why this exists

Content moderation at scale is expensive. Off-the-shelf APIs (Perspective API, OpenAI Moderation) cost per-request and offer no customization. This project explores whether a small fine-tuned model can match their performance for a fraction of the cost.

## Running it

Requires Python 3.11+.

​bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python explore_data.py
​