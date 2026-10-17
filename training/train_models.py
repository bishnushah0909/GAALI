#!/usr/bin/env python3
# ============================================================
# FINAL EXPERIMENT
# OFFENSIVE vs NON-OFFENSIVE CLASSIFICATION
#
# Primary metric:
#       Macro F1
#
# Secondary metric:
#       Weighted F1
#
# Models:
#       1. SVC (Linear)
#       2. Multinomial Naive Bayes
#       3. mBERT
#       4. XLM-RoBERTa
#       5. MuRIL
#       6. IndicBERT
#
# Dataset:
#       Stratified 80:10:10 train/dev/test split
#
# IMPORTANT:
#       Transformer checkpoint selection = Development Macro F1
#       Final model evaluation = Unseen Test Set
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import gc
import random
import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report
)

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 90)
print("FINAL EXPERIMENT: MACRO F1 PRIMARY / WEIGHTED F1 SECONDARY")
print("=" * 90)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# 3. DATASET PATH (Supports both Kaggle and local repository)
# ============================================================

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

FINAL_RESULTS_FILE = os.path.join(
    _WORK_DIR, "cleaned_final_results_macro_primary.csv"
)

BASELINE_RESULTS_FILE = os.path.join(
    _WORK_DIR, "cleaned_baseline_results_macro_primary.csv"
)

TRANSFORMER_RESULTS_FILE = os.path.join(
    _WORK_DIR, "cleaned_transformer_results_macro_primary.csv"
)

PREDICTIONS_FILE = os.path.join(
    _LOCAL_DATA_DIR, "test_set_model_predictions.csv"
)


# ============================================================
# 4. CHECK DATASET PATH
# ============================================================

print("\nChecking dataset path...")
print(DATA_PATH)

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        "\nDataset was not found at:\n"
        f"{DATA_PATH}\n\n"
        "Please verify the Kaggle dataset attachment or local data/ folder."
    )

print("Dataset found successfully.")


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")


# ============================================================
# 6. VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = {
    "Comment",
    "Label",
    "split"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:

    raise ValueError(
        f"The following required columns are missing: "
        f"{sorted(missing_columns)}"
    )

print("\nRequired columns verified:")
print(sorted(required_columns))


# ============================================================
# 7. DATA VALIDATION
# ============================================================

# Make sure labels are integers
df["Label"] = df["Label"].astype(int)

# Ensure comments are strings
df["Comment"] = df["Comment"].astype(str)

# Check missing values
missing_comments = df["Comment"].isna().sum()
missing_labels = df["Label"].isna().sum()
missing_splits = df["split"].isna().sum()

if missing_comments > 0 or missing_labels > 0 or missing_splits > 0:

    raise ValueError(
        "Missing values detected in Comment, Label, or split column. "
        "Please correct the dataset before training."
    )

# Verify labels
unique_labels = sorted(df["Label"].unique().tolist())

if unique_labels != [0, 1]:

    raise ValueError(
        f"Expected binary labels [0, 1], but found: {unique_labels}"
    )

# Verify split names
valid_splits = {"train", "dev", "test"}

found_splits = set(df["split"].unique())

if found_splits != valid_splits:

    raise ValueError(
        f"Expected split values {valid_splits}, "
        f"but found: {found_splits}"
    )

print("\nDataset validation passed.")


# ============================================================
# 8. CREATE TRAIN / DEV / TEST SUBSETS
# ============================================================

train_df = df[df["split"] == "train"].copy()
dev_df = df[df["split"] == "dev"].copy()
test_df = df[df["split"] == "test"].copy()

print("\n" + "=" * 90)
print("DATASET PARTITION")
print("=" * 90)

print(f"Training set:   {len(train_df)}")
print(f"Development:    {len(dev_df)}")
print(f"Test set:       {len(test_df)}")
print(f"Total:          {len(df)}")


# ============================================================
# 9. VERIFY PARTITION SIZES
# ============================================================

total = len(df)

train_percentage = len(train_df) / total * 100
dev_percentage = len(dev_df) / total * 100
test_percentage = len(test_df) / total * 100

print("\nPartition percentages:")

print(f"Train: {train_percentage:.2f}%")
print(f"Dev:   {dev_percentage:.2f}%")
print(f"Test:  {test_percentage:.2f}%")


# ============================================================
# 10. VERIFY CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(data, name):

    distribution = (
        data["Label"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
    )

    print(f"\n{name} class distribution:")

    for label, percentage in distribution.items():

        label_name = (
            "Non-offensive"
            if label == 0
            else "Offensive"
        )

        print(
            f"  {label} ({label_name}): "
            f"{percentage:.2f}%"
        )


print_class_distribution(df, "Overall dataset")
print_class_distribution(train_df, "Training set")
print_class_distribution(dev_df, "Development set")
print_class_distribution(test_df, "Test set")


# ============================================================
# 11. COMMON METRIC FUNCTION
# ============================================================

def calculate_all_metrics(y_true, y_pred):
    """
    Calculate:
        Macro Precision
        Macro Recall
        Macro F1
        Weighted Precision
        Weighted Recall
        Weighted F1
        Accuracy
    """

    # -----------------------------
    # Macro metrics
    # -----------------------------

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    # -----------------------------
    # Weighted metrics
    # -----------------------------

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    # -----------------------------
    # Accuracy
    # -----------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    return {
        "Macro Precision": round(float(macro_precision), 4),
        "Macro Recall": round(float(macro_recall), 4),
        "Macro F1": round(float(macro_f1), 4),

        "Weighted Precision": round(float(weighted_precision), 4),
        "Weighted Recall": round(float(weighted_recall), 4),
        "Weighted F1": round(float(weighted_f1), 4),

        "Accuracy": round(float(accuracy), 4)
    }


# ============================================================
# 12. INITIALIZE RESULT STORAGE
# ============================================================

final_results = []

baseline_results = []
transformer_results = []

test_predictions_df = pd.DataFrame({
    "Comment": test_df["Comment"].values,
    "true_label": test_df["Label"].values,
})


# ============================================================
# 13. CLASSICAL MACHINE-LEARNING BASELINES
# ============================================================

print("\n\n")
print("=" * 90)
print("CLASSICAL MACHINE-LEARNING BASELINES")
print("=" * 90)


# ------------------------------------------------------------
# 13.1 Prepare baseline data
# ------------------------------------------------------------

X_train = train_df["Comment"]
y_train = train_df["Label"]

X_test = test_df["Comment"]
y_test = test_df["Label"]


# ------------------------------------------------------------
# 13.2 TF-IDF
# ------------------------------------------------------------

print("\nFitting TF-IDF vectorizer on TRAINING DATA ONLY...")

vectorizer = TfidfVectorizer(
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(
    X_train
)

X_test_vec = vectorizer.transform(
    X_test
)

print(
    f"Training TF-IDF matrix: "
    f"{X_train_vec.shape}"
)

print(
    f"Test TF-IDF matrix:     "
    f"{X_test_vec.shape}"
)


# ============================================================
# 14. SVC WITH LINEAR KERNEL
# ============================================================

print("\n" + "-" * 70)
print("Training SVC with linear kernel...")
print("-" * 70)

svc = SVC(
    kernel="linear",
    random_state=SEED
)

svc.fit(
    X_train_vec,
    y_train
)

svc_predictions = svc.predict(
    X_test_vec
)

test_predictions_df["svc_pred"] = svc_predictions

svc_metrics = calculate_all_metrics(
    y_test,
    svc_predictions
)

svc_result = {
    "Model": "SVC (Linear)",
    **svc_metrics
}

baseline_results.append(
    svc_result
)

final_results.append(
    svc_result
)

print("\nSVC (Linear) classification report:")
print(
    classification_report(
        y_test,
        svc_predictions,
        digits=4,
        zero_division=0
    )
)

print("\nSVC (Linear) summary:")

for key, value in svc_result.items():
    print(f"{key}: {value}")


# ============================================================
# 15. MULTINOMIAL NAIVE BAYES
# ============================================================

print("\n" + "-" * 70)
print("Training Multinomial Naive Bayes...")
print("-" * 70)

mnb = MultinomialNB()

mnb.fit(
    X_train_vec,
    y_train
)

mnb_predictions = mnb.predict(
    X_test_vec
)

test_predictions_df["mnb_pred"] = mnb_predictions

mnb_metrics = calculate_all_metrics(
    y_test,
    mnb_predictions
)

mnb_result = {
    "Model": "Multinomial Naive Bayes",
    **mnb_metrics
}

baseline_results.append(
    mnb_result
)

final_results.append(
    mnb_result
)

print("\nMultinomial Naive Bayes classification report:")

print(
    classification_report(
        y_test,
        mnb_predictions,
        digits=4,
        zero_division=0
    )
)

print("\nMultinomial Naive Bayes summary:")

for key, value in mnb_result.items():
    print(f"{key}: {value}")


# ============================================================
# 16. SAVE BASELINE RESULTS
# ============================================================

baseline_results_df = pd.DataFrame(
    baseline_results
)

baseline_results_df.to_csv(
    BASELINE_RESULTS_FILE,
    index=False
)

print(
    f"\nClassical baseline results saved to:\n"
    f"{BASELINE_RESULTS_FILE}"
)


# ============================================================
# 17. CLEAN CLASSICAL MODEL MEMORY
# ============================================================

del vectorizer
del X_train_vec
del X_test_vec
del svc
del mnb

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 18. TRANSFORMER MODELS
# ============================================================

models_to_train = [

    {
        "name": "mBERT",
        "repo": "bert-base-multilingual-cased",
        "pred_col": "mbert_pred"
    },

    {
        "name": "XLM-RoBERTa",
        "repo": "xlm-roberta-base",
        "pred_col": "xlm_roberta_pred"
    },

    {
        "name": "MuRIL",
        "repo": "google/muril-base-cased",
        "pred_col": "muril_pred"
    },

    {
        "name": "IndicBERT",
        "repo": "ai4bharat/indic-bert",
        "pred_col": "indicbert_pred"
    }
]


# ============================================================
# 19. TRANSFORMER METRIC FUNCTION
# ============================================================

def compute_transformer_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    # Macro
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    # Weighted
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

        # PRIMARY MODEL-SELECTION METRIC
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,

        # SECONDARY METRICS
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,

        "accuracy": accuracy
    }


# ============================================================
# 20. TRANSFORMER TRAINING LOOP
# ============================================================

for item in models_to_train:

    model_name = item["name"]
    repo_id = item["repo"]
    pred_col = item["pred_col"]

    print("\n\n")
    print("=" * 90)
    print(f"STARTING TRANSFORMER: {model_name}")
    print(f"Model checkpoint: {repo_id}")
    print("=" * 90)

    model = None
    tokenizer = None
    trainer = None

    try:

        # ====================================================
        # 20.1 LOAD TOKENIZER
        # ====================================================

        print(f"\nLoading tokenizer for {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(
            repo_id
        )


        # ====================================================
        # 20.2 LOAD MODEL
        # ====================================================

        print(f"Loading model for {model_name}...")

        model = AutoModelForSequenceClassification.from_pretrained(
            repo_id,
            num_labels=2
        )


        # ====================================================
        # 20.3 CREATE DATASETS
        # ====================================================

        ds_train = Dataset.from_pandas(
            train_df[
                ["Comment", "Label"]
            ],
            preserve_index=False
        )

        ds_dev = Dataset.from_pandas(
            dev_df[
                ["Comment", "Label"]
            ],
            preserve_index=False
        )

        ds_test = Dataset.from_pandas(
            test_df[
                ["Comment", "Label"]
            ],
            preserve_index=False
        )


        # ====================================================
        # 20.4 RENAME LABEL COLUMN
        # ====================================================

        ds_train = ds_train.rename_column(
            "Label",
            "labels"
        )

        ds_dev = ds_dev.rename_column(
            "Label",
            "labels"
        )

        ds_test = ds_test.rename_column(
            "Label",
            "labels"
        )


        # ====================================================
        # 20.5 TOKENIZATION
        # ====================================================

        def tokenize_function(examples):

            return tokenizer(
                examples["Comment"],
                truncation=True,
                max_length=128
            )


        print(f"Tokenizing {model_name} datasets...")

        ds_train = ds_train.map(
            tokenize_function,
            batched=True
        )

        ds_dev = ds_dev.map(
            tokenize_function,
            batched=True
        )

        ds_test = ds_test.map(
            tokenize_function,
            batched=True
        )


        # ====================================================
        # 20.6 DYNAMIC PADDING
        # ====================================================

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer
        )


        # ====================================================
        # 20.7 TRAINING ARGUMENTS
        # ====================================================

        output_dir = os.path.join(
            _WORK_DIR, f"results_{model_name}"
        )

        training_args = TrainingArguments(

            output_dir=output_dir,

            # ----------------------------------------------
            # Evaluation
            # ----------------------------------------------

            eval_strategy="epoch",

            # ----------------------------------------------
            # Checkpoint saving
            # ----------------------------------------------

            save_strategy="epoch",
            save_total_limit=1,

            # ----------------------------------------------
            # Optimization
            # ----------------------------------------------

            learning_rate=2e-5,
            weight_decay=0.01,

            # ----------------------------------------------
            # Batch sizes
            # ----------------------------------------------

            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,

            # ----------------------------------------------
            # Training duration
            # ----------------------------------------------

            num_train_epochs=3,

            # ----------------------------------------------
            # Reproducibility
            # ----------------------------------------------

            seed=SEED,
            data_seed=SEED,

            # ----------------------------------------------
            # BEST CHECKPOINT SELECTION
            #
            # PRIMARY METRIC = MACRO F1
            # ----------------------------------------------

            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,

            # ----------------------------------------------
            # Mixed precision
            # ----------------------------------------------

            fp16=torch.cuda.is_available(),

            # ----------------------------------------------
            # Logging
            # ----------------------------------------------

            logging_steps=50,
            report_to="none"
        )


        # ====================================================
        # 20.8 INITIALIZE TRAINER
        # ====================================================

        trainer = Trainer(

            model=model,

            args=training_args,

            train_dataset=ds_train,

            eval_dataset=ds_dev,

            processing_class=tokenizer,

            data_collator=data_collator,

            compute_metrics=compute_transformer_metrics
        )


        # ====================================================
        # 20.9 TRAIN MODEL
        # ====================================================

        print(
            f"\nFine-tuning {model_name}..."
        )

        trainer.train()


        # ====================================================
        # 20.10 DEVELOPMENT SET EVALUATION
        # ====================================================

        print(
            f"\nEvaluating {model_name} "
            f"on DEVELOPMENT set..."
        )

        dev_results = trainer.evaluate(
            eval_dataset=ds_dev
        )

        development_macro_f1 = (
            dev_results["eval_macro_f1"]
        )

        development_weighted_f1 = (
            dev_results["eval_weighted_f1"]
        )

        print(
            f"Development Macro F1: "
            f"{development_macro_f1:.4f}"
        )

        print(
            f"Development Weighted F1: "
            f"{development_weighted_f1:.4f}"
        )


        # ====================================================
        # 20.11 FINAL TEST EVALUATION
        # ====================================================

        print(
            f"\nEvaluating {model_name} "
            f"on UNSEEN TEST set..."
        )

        test_pred_out = trainer.predict(ds_test)
        test_results = test_pred_out.metrics
        test_predictions_df[pred_col] = np.argmax(
            test_pred_out.predictions, axis=-1
        )


        # ====================================================
        # 20.12 EXTRACT FINAL TEST METRICS
        # ====================================================

        transformer_result = {

            "Model": model_name,

            "Macro Precision": round(
                float(
                    test_results[
                        "test_macro_precision"
                    ]
                ),
                4
            ),

            "Macro Recall": round(
                float(
                    test_results[
                        "test_macro_recall"
                    ]
                ),
                4
            ),

            "Macro F1": round(
                float(
                    test_results[
                        "test_macro_f1"
                    ]
                ),
                4
            ),

            "Weighted Precision": round(
                float(
                    test_results[
                        "test_weighted_precision"
                    ]
                ),
                4
            ),

            "Weighted Recall": round(
                float(
                    test_results[
                        "test_weighted_recall"
                    ]
                ),
                4
            ),

            "Weighted F1": round(
                float(
                    test_results[
                        "test_weighted_f1"
                    ]
                ),
                4
            ),

            "Accuracy": round(
                float(
                    test_results[
                        "test_accuracy"
                    ]
                ),
                4
            )
        }


        # ====================================================
        # 20.13 STORE RESULTS
        # ====================================================

        transformer_results.append(
            transformer_result
        )

        final_results.append(
            transformer_result
        )


        # ====================================================
        # 20.14 DISPLAY RESULTS
        # ====================================================

        print("\n" + "-" * 70)
        print(f"{model_name} FINAL TEST RESULTS")
        print("-" * 70)

        for key, value in transformer_result.items():

            print(
                f"{key}: {value}"
            )

        print("-" * 70)


        # ====================================================
        # 20.15 SAVE RESULTS AFTER EVERY MODEL
        # ====================================================

        current_transformer_df = pd.DataFrame(
            transformer_results
        )

        current_transformer_df.to_csv(
            TRANSFORMER_RESULTS_FILE,
            index=False
        )


        current_final_df = pd.DataFrame(
            final_results
        )

        current_final_df.to_csv(
            FINAL_RESULTS_FILE,
            index=False
        )

        test_predictions_df.to_csv(
            PREDICTIONS_FILE,
            index=False
        )

        print(
            "\nResults successfully saved."
        )


    except Exception as e:

        print("\n")
        print("!" * 90)
        print(f"ERROR WHILE TRAINING {model_name}")
        print("!" * 90)
        print(str(e))
        print("!" * 90)

        # Continue to the next model
        continue


    finally:

        # ====================================================
        # CLEAN MEMORY
        # ====================================================

        if model is not None:
            del model

        if tokenizer is not None:
            del tokenizer

        if trainer is not None:
            del trainer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"\nMemory cleanup completed for {model_name}."
        )


# ============================================================
# 21. FINAL RESULTS DATAFRAME
# ============================================================

final_results_df = pd.DataFrame(
    final_results
)


# ============================================================
# 22. ENFORCE MODEL ORDER
# ============================================================

model_order = [

    "SVC (Linear)",

    "Multinomial Naive Bayes",

    "mBERT",

    "XLM-RoBERTa",

    "MuRIL",

    "IndicBERT"
]


final_results_df["Model"] = pd.Categorical(

    final_results_df["Model"],

    categories=model_order,

    ordered=True
)


final_results_df = (

    final_results_df

    .sort_values("Model")

    .reset_index(drop=True)
)


# ============================================================
# 23. SAVE FINAL RESULTS
# ============================================================

final_results_df.to_csv(
    FINAL_RESULTS_FILE,
    index=False
)


# ============================================================
# 24. DISPLAY FINAL RESULTS
# ============================================================

print("\n\n")
print("=" * 110)
print("FINAL MODEL COMPARISON")
print("=" * 110)

print(
    final_results_df.to_string(
        index=False
    )
)


# ============================================================
# 25. PRIMARY MODEL RANKING
# ============================================================

print("\n\n")
print("=" * 90)
print("MODEL RANKING BY PRIMARY METRIC: MACRO F1")
print("=" * 90)

ranking_df = (

    final_results_df[
        [
            "Model",
            "Macro F1",
            "Weighted F1",
            "Accuracy"
        ]
    ]

    .sort_values(
        "Macro F1",
        ascending=False
    )

    .reset_index(drop=True)
)


print(
    ranking_df.to_string(
        index=False
    )
)


# ============================================================
# 26. RESEARCH-PAPER MARKDOWN TABLE
# ============================================================

print("\n\n")
print("=" * 110)
print("TABLE FOR RESEARCH PAPER")
print("=" * 110)

print(
    final_results_df.to_markdown(
        index=False
    )
)


# ============================================================
# 27. FINAL FILE LOCATIONS
# ============================================================

print("\n\n")
print("=" * 90)
print("EXPERIMENT COMPLETED")
print("=" * 90)

print(
    "\nFinal combined results:"
)

print(
    FINAL_RESULTS_FILE
)

print(
    "\nClassical baseline results:"
)

print(
    BASELINE_RESULTS_FILE
)

print(
    "\nTransformer results:"
)

print(
    TRANSFORMER_RESULTS_FILE
)

print("\n" + "=" * 90)
