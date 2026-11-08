# DistilBERT NER

This experimental NER pipeline fine-tunes DistilBERT on the existing BIO files in `nlp/data/entities/augmented v2/bio`. It saves the fine-tuned model under `nlp/models/ner/distilbert` and writes held-out evaluation metrics to `nlp/results/ner/distilbert-ner.txt`.

## Setup

Install the NLP requirements into the project environment:

```powershell
.\venv\Scripts\python.exe -m pip install -r nlp_requirements.txt
```

The initial training run needs access to `distilbert/distilbert-base-uncased`, unless that model is already cached locally.

## Train and evaluate

From the project root:

```powershell
.\venv\Scripts\python.exe nlp\train\train_ner_distilbert.py
```

Training uses the same deterministic 80/10/10 sentence split as the BiLSTM script. It selects the saved checkpoint by validation `PATH_FILE` F1, then evaluates that checkpoint on the held-out test set.

For a faster CPU-only preliminary run, use fewer epochs and freeze lower layers:

```powershell
.\venv\Scripts\python.exe nlp\train\train_ner_distilbert.py --epochs 2 --batch-size 32 --freeze-layers 4
```

This reduces training work but can lower the final score; use full fine-tuning on a CUDA-enabled machine for the stronger comparison.

For a CUDA GPU with limited VRAM, use a small batch with gradient accumulation:

```powershell
.\venv\Scripts\python.exe nlp\train\train_ner_distilbert.py --epochs 3 --batch-size 2 --gradient-accumulation-steps 8 --freeze-layers 2
```

## Try the trained model

```powershell
.\venv\Scripts\python.exe nlp\scripts\ner\classifier-ner-distilbert.py
```

The current WinPilot command parser does not yet call this experimental NER model. Compare its test metrics and inference latency with the BiLSTM before wiring NER predictions into command execution.
