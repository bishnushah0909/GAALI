#!/usr/bin/env python3
# ============================================================
# INDICBERT FINAL RUN (With Gated Hugging Face Login)
#
# Primary metric:
#      Macro F1
#
# Secondary metric:
#      Weighted F1
#
# Best checkpoint:
#      Highest Development Macro F1
#
# Final evaluation:
#      Unseen Test Set
# ============================================================

import os
import gc
import random
from getpass import getpass
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from huggingface_hub import login

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)


# ------------------------------------------------------------
# 0. Hugging Face Authentication (required for gated ai4bharat/indic-bert)
# ------------------------------------------------------------

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("Enter your NEW Hugging Face token.")
    print("The token will not be displayed.")
    hf_token = getpass("Hugging Face token: ")

if not hf_token:
    raise ValueError("No Hugging Face token was entered.")

login(token=hf_token)
print("Hugging Face login completed.")


# ------------------------------------------------------------
# 1. Reproducibility
# ------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 90)
print("INDICBERT FINAL EXPERIMENT")
print("=" * 90)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ------------------------------------------------------------
# 2. Exact dataset path (Supports both Kaggle and local repository)
# ------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_DATA_DIR = os.environ.get("GAALI_DATA_DIR", os.path.join(_REPO_ROOT, "data"))
_KAGGLE_DATA_PATH = (
    "/kaggle/input/datasets/bishnucs/finalexp-on28thaug/"
    "corrected_cleaned_combined_data_kaggle_stratified.csv"
)

DATA_PATH = (
    _KAGGLE_DATA_PATH
    if os.path.exists(_KAGGLE_DATA_PATH)
    else os.path.join(_LOCAL_DATA_DIR, "corrected_cleaned_combined_data_kaggle_stratified.csv")
)

_WORK_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else _LOCAL_DATA_DIR

OUTPUT_DIR = os.path.join(
    _WORK_DIR, "results_IndicBERT_macro_final"
)

RESULT_FILE = os.path.join(
    _WORK_DIR, "IndicBERT_final_results_macro_primary.csv"
)

PREDICTIONS_FILE = os.path.join(
    _LOCAL_DATA_DIR, "test_set_model_predictions.csv"
)


# ------------------------------------------------------------
# 3. Check dataset
# ------------------------------------------------------------

print("\nChecking dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {len(df)} rows")


# ------------------------------------------------------------
# 4. Validate columns
# ------------------------------------------------------------

required_columns = {
    "Comment",
    "Label",
    "split"
}

missing_columns = (
    required_columns - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing columns: {sorted(missing_columns)}"
    )


# ------------------------------------------------------------
# 5. Validate data
# ------------------------------------------------------------

if df[["Comment", "Label", "split"]].isnull().any().any():
    raise ValueError(
        "Missing values found in Comment, Label, or split."
    )

df["Comment"] = df["Comment"].astype(str)
df["Label"] = df["Label"].astype(int)

if sorted(df["Label"].unique()) != [0, 1]:
    raise ValueError(
        f"Expected labels [0, 1], found: "
        f"{sorted(df['Label'].unique())}"
    )

expected_splits = {"train", "dev", "test"}
actual_splits = set(df["split"].unique())

if actual_splits != expected_splits:
    raise ValueError(
        f"Expected splits {expected_splits}, "
        f"found {actual_splits}"
    )


# ------------------------------------------------------------
# 6. Extract existing split
# ------------------------------------------------------------

train_df = df[df["split"] == "train"].copy()
dev_df = df[df["split"] == "dev"].copy()
test_df = df[df["split"] == "test"].copy()

print("\nDataset partition:")
print(f"Train: {len(train_df)}")
print(f"Dev:   {len(dev_df)}")
print(f"Test:  {len(test_df)}")
print(
    f"Total: {len(train_df) + len(dev_df) + len(test_df)}"
)


# ------------------------------------------------------------
# 7. Load IndicBERT
# ------------------------------------------------------------

MODEL_NAME = "IndicBERT"
MODEL_REPO = "ai4bharat/indic-bert"

print("\n" + "=" * 90)
print(f"Loading {MODEL_NAME}")
print(f"Repository: {MODEL_REPO}")
print("=" * 90)

try:

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_REPO,
        num_labels=2
    )

except Exception as e:

    raise RuntimeError(
        "\nIndicBERT could not be loaded.\n\n"
        "Check that:\n"
        "1. You completed Hugging Face login.\n"
        "2. Your account has access to the gated IndicBERT model.\n"
        "3. You accepted the model access conditions.\n"
        "4. Kaggle Internet is enabled.\n\n"
        f"Original error:\n{e}"
    ) from e

print("IndicBERT loaded successfully.")


# ------------------------------------------------------------
# 8. Tokenization
# ------------------------------------------------------------

def tokenize_function(examples):

    return tokenizer(
        examples["Comment"],
        truncation=True,
        max_length=128
    )


# ------------------------------------------------------------
# 9. Create Hugging Face datasets
# ------------------------------------------------------------

ds_train = Dataset.from_pandas(
    train_df[["Comment", "Label"]],
    preserve_index=False
)

ds_dev = Dataset.from_pandas(
    dev_df[["Comment", "Label"]],
    preserve_index=False
)

ds_test = Dataset.from_pandas(
    test_df[["Comment", "Label"]],
    preserve_index=False
)


# Rename target column
ds_train = ds_train.rename_column("Label", "labels")
ds_dev = ds_dev.rename_column("Label", "labels")
ds_test = ds_test.rename_column("Label", "labels")


# ------------------------------------------------------------
# 10. Tokenize
# ------------------------------------------------------------

print("\nTokenizing training data...")
ds_train = ds_train.map(
    tokenize_function,
    batched=True
)

print("Tokenizing development data...")
ds_dev = ds_dev.map(
    tokenize_function,
    batched=True
)

print("Tokenizing test data...")
ds_test = ds_test.map(
    tokenize_function,
    batched=True
)

print("Tokenization completed.")


# ------------------------------------------------------------
# 11. Dynamic padding
# ------------------------------------------------------------

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ------------------------------------------------------------
# 12. Metrics
# ------------------------------------------------------------

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    # Macro metrics
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    # Weighted metrics
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    # Accuracy
    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,

        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,

        "accuracy": accuracy
    }


# ------------------------------------------------------------
# 13. Training configuration
# ------------------------------------------------------------

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    # Evaluate after every epoch
    eval_strategy="epoch",

    # Save after every epoch
    save_strategy="epoch",

    save_total_limit=1,

    # Same settings as the other transformer experiments
    learning_rate=2e-5,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=32,

    num_train_epochs=3,

    weight_decay=0.01,

    # Reproducibility
    seed=SEED,
    data_seed=SEED,

    # PRIMARY CHECKPOINT SELECTION METRIC
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,

    # FP16 when GPU is available
    fp16=torch.cuda.is_available(),

    logging_steps=50,

    report_to="none"
)


# ------------------------------------------------------------
# 14. Trainer
# ------------------------------------------------------------

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=ds_train,

    eval_dataset=ds_dev,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ------------------------------------------------------------
# 15. Train
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("STARTING INDICBERT TRAINING")
print("=" * 90)

trainer.train()


# ------------------------------------------------------------
# 16. Development evaluation
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("INDICBERT DEVELOPMENT RESULTS")
print("=" * 90)

dev_results = trainer.evaluate(
    eval_dataset=ds_dev
)

print(
    f"Macro Precision: "
    f"{dev_results['eval_macro_precision']:.4f}"
)

print(
    f"Macro Recall: "
    f"{dev_results['eval_macro_recall']:.4f}"
)

print(
    f"Macro F1: "
    f"{dev_results['eval_macro_f1']:.4f}"
)

print(
    f"Weighted Precision: "
    f"{dev_results['eval_weighted_precision']:.4f}"
)

print(
    f"Weighted Recall: "
    f"{dev_results['eval_weighted_recall']:.4f}"
)

print(
    f"Weighted F1: "
    f"{dev_results['eval_weighted_f1']:.4f}"
)

print(
    f"Accuracy: "
    f"{dev_results['eval_accuracy']:.4f}"
)


# ------------------------------------------------------------
# 17. Final TEST evaluation & prediction export
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("INDICBERT FINAL TEST RESULTS")
print("=" * 90)

test_pred_out = trainer.predict(ds_test)
test_results = test_pred_out.metrics
indicbert_preds = np.argmax(test_pred_out.predictions, axis=-1)


# ------------------------------------------------------------
# 18. Extract final result
# ------------------------------------------------------------

indicbert_result = {

    "Model": "IndicBERT",

    "Macro Precision": round(
        float(test_results["test_macro_precision"]),
        4
    ),

    "Macro Recall": round(
        float(test_results["test_macro_recall"]),
        4
    ),

    "Macro F1": round(
        float(test_results["test_macro_f1"]),
        4
    ),

    "Weighted Precision": round(
        float(test_results["test_weighted_precision"]),
        4
    ),

    "Weighted Recall": round(
        float(test_results["test_weighted_recall"]),
        4
    ),

    "Weighted F1": round(
        float(test_results["test_weighted_f1"]),
        4
    ),

    "Accuracy": round(
        float(test_results["test_accuracy"]),
        4
    )
}


# ------------------------------------------------------------
# 19. Print final result
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("INDICBERT FINAL SCORE")
print("=" * 90)

for key, value in indicbert_result.items():
    print(f"{key}: {value}")


# ------------------------------------------------------------
# 20. Save final result and update test_set_model_predictions.csv
# ------------------------------------------------------------

result_df = pd.DataFrame(
    [indicbert_result]
)

result_df.to_csv(
    RESULT_FILE,
    index=False
)

if os.path.exists(PREDICTIONS_FILE):
    pred_df = pd.read_csv(PREDICTIONS_FILE)
    pred_df["indicbert_pred"] = indicbert_preds
    pred_df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"\nUpdated indicbert_pred column in:\n{PREDICTIONS_FILE}")

print("\nResult saved to:")
print(RESULT_FILE)


# ------------------------------------------------------------
# 21. Cleanup
# ------------------------------------------------------------

del model
del trainer
del tokenizer
del ds_train
del ds_dev
del ds_test

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "=" * 90)
print("INDICBERT RUN COMPLETED SUCCESSFULLY")
print("=" * 90)
