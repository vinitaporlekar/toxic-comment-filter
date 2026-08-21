## Day 3 — Understanding DistilBERT + Colab Setup ✓

**What I did:**
- Learned what DistilBERT is and why we're using it
- Learned tokenization: subword vocab, [CLS]/[SEP]/[PAD], attention masks
- Set up Google Colab with T4 GPU (verified via torch.cuda.is_available())
- Ran the tokenizer locally on real Jigsaw comments

**Key concepts:**
- DistilBERT = compressed BERT (60% size, 97% accuracy)
- Pre-trained models already know English; fine-tuning teaches them our task
- Text → subword tokens → numeric IDs → neural network
- Max length 256 tokens covers 75% of our comments fully
- Attention mask distinguishes real tokens from padding

**Setup verified:**
- Colab GPU: Tesla T4, 15.8 GB memory ✓
- Transformers library installed locally ✓
- Can tokenize Jigsaw comments successfully ✓

**Blocked on:**
- Nothing
