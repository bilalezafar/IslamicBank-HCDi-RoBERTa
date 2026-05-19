# IslamicBank-HCDi-RoBERTa

**IslamicBank-HCDi-RoBERTa** is an NLP transformer model for classifying dimensions of human capital disclosure in Islamic banking. The project includes the training notebook, labelled dataset, model comparison outputs, evaluation results, and figures used to fine-tune and evaluate transformer models for seven-class human capital disclosure classification.

## Project purpose

This repository supports research on human capital disclosure in Islamic banking. It provides reproducible materials for training and evaluating transformer-based classifiers that identify whether a text segment is related to human capital and assign it to one of six disclosure dimensions or to a separate `Not related` class.

## Classification and Dataset

The dataset used in this repository is `hc_sentences_bert.csv`.

The final labelled dataset contains **13,218 observations** across seven classes.

| Classification | Number of observations |
|---|---:|
| Not related | 2,500 |
| Diversity, Equity, and Inclusion | 1,106 |
| Compensation and Benefits | 2,000 |
| Labour Relations and Organisational Culture | 2,000 |
| Health, Safety, and Well-being | 1,731 |
| Workforce Demographics | 1,881 |
| Training, Development, and Human Capital Investment | 2,000 |

The dataset was split using a stratified 80:10:10 train-validation-test design.

| Split | Number of observations |
|---|---:|
| Training | 10,574 |
| Validation | 1,322 |
| Test | 1,322 |

## Models compared

Four transformer models were fine-tuned and compared:

| Model | Hugging Face ID |
|---|---|
| BERT base | `google-bert/bert-base-uncased` |
| RoBERTa base | `FacebookAI/roberta-base` |
| DeBERTa V3 base | `microsoft/deberta-v3-base` |
| FinBERT pretrain | `yiyanghkust/finbert-pretrain` |

## Best model

The best-performing model was **RoBERTa base**, saved on Hugging Face as:

[bilalzafar/IslamicBank-HCDi-RoBERTa](https://huggingface.co/bilalzafar/IslamicBank-HCDi-RoBERTa)

RoBERTa was selected based on validation macro-F1.

| Metric | Validation | Test |
|---|---:|---:|
| Macro-F1 | 0.895 | 0.890 |
| Weighted-F1 | 0.891 | 0.888 |
| Accuracy | 0.891 | 0.888 |

## Model comparison results

| Model | Validation Macro-F1 | Test Macro-F1 | Test Accuracy |
|---|---:|---:|---:|
| RoBERTa base | 0.895 | 0.890 | 0.888 |
| BERT base | 0.894 | 0.883 | 0.881 |
| DeBERTa V3 base | 0.855 | 0.848 | 0.845 |
| FinBERT pretrain | 0.841 | 0.852 | 0.849 |

## Class-wise test performance of RoBERTa

| Label ID | Classification | Precision | Recall | F1-score | Support |
|---:|---|---:|---:|---:|---:|
| 0 | Not related | 0.888 | 0.852 | 0.869 | 250 |
| 1 | Diversity, Equity, and Inclusion | 0.917 | 0.901 | 0.909 | 111 |
| 2 | Compensation and Benefits | 0.914 | 0.955 | 0.934 | 200 |
| 3 | Labour Relations and Organisational Culture | 0.841 | 0.795 | 0.817 | 200 |
| 4 | Health, Safety, and Well-being | 0.925 | 0.925 | 0.925 | 173 |
| 5 | Workforce Demographics | 0.881 | 0.904 | 0.892 | 188 |
| 6 | Training, Development, and Human Capital Investment | 0.866 | 0.905 | 0.885 | 200 |

## Repository structure

```text
IslamicBank-HCDi-RoBERTa/
│
├── README.md
├── hc_dimensions.ipynb
├── hc_sentences_bert.csv
│
└── outputs/
    ├── model_comparison_summary.csv
    ├── classwise_performance_all_models.csv
    ├── label_mapping.csv
    ├── label_mapping.json
    ├── class_weights.csv
    ├── train_split.csv
    ├── validation_split.csv
    ├── test_split.csv
    ├── best_model_summary.csv
    ├── classification_report_*.csv
    ├── confusion_matrix_*.csv
    ├── test_predictions_*.csv
    ├── trainer_log_history_*.csv
    └── *.svg
````

## Key output files

| File                                             | Description                                                        |
| ------------------------------------------------ | ------------------------------------------------------------------ |
| `hc_dimensions.ipynb`                            | Main notebook for training and evaluation                          |
| `hc_sentences_bert.csv`                          | Labelled dataset used for model training                           |
| `outputs/model_comparison_summary.csv`           | Summary of all model-level results                                 |
| `outputs/classwise_performance_all_models.csv`   | Class-wise precision, recall and F1-score for all models           |
| `outputs/classification_report_roberta_base.csv` | Classification report for the selected RoBERTa model               |
| `outputs/test_predictions_roberta_base.csv`      | Test-set predictions and class probabilities for RoBERTa           |
| `outputs/confusion_matrix_roberta_base.csv`      | Confusion matrix for RoBERTa                                       |
| `outputs/best_model_summary.csv`                 | Summary of the selected best model                                 |
| `outputs/*.svg`                                  | Figures for model comparison, class-wise F1 and confusion matrices |

## How to use the Hugging Face model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_id = "bilalzafar/IslamicBank-HCDi-RoBERTa"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,
    truncation=True,
    max_length=128
)

text = "The Bank provides regular training programmes to enhance employees' professional and Shariah-related competencies."

predictions = classifier(text)
print(predictions)
```

## Suggested two-stage workflow

For large-scale human capital disclosure analysis, the following workflow is recommended:

1. Use [bilalzafar/HC-BERT-IFIs](https://huggingface.co/bilalzafar/HC-BERT-IFIs) to classify text into `HC` and `Non-HC`.
2. Use [bilalzafar/IslamicBank-HCDi-RoBERTa](https://huggingface.co/bilalzafar/IslamicBank-HCDi-RoBERTa) to classify HC-related text into human capital disclosure dimensions.

## Training configuration

| Parameter                     | Value                                      |
| ----------------------------- | ------------------------------------------ |
| Task                          | Supervised multi-class text classification |
| Base model of best classifier | `FacebookAI/roberta-base`                  |
| Number of labels              | 7                                          |
| Maximum sequence length       | 128                                        |
| Epochs                        | 5                                          |
| Learning rate                 | 2e-5                                       |
| Weight decay                  | 0.01                                       |
| Loss function                 | Class-weighted cross-entropy               |
| Selection metric              | Validation macro-F1                        |
| Hardware                      | NVIDIA Tesla T4 GPU                        |


## Citation

If you use this repository, model, dataset, outputs or replication materials, please cite:

```bibtex
@misc{zafar2026_islamicbank_hcdi_roberta,
  title        = {IslamicBank-HCDi-RoBERTa: A Transformer Model for Classifying Dimensions of Human Capital Disclosure in Islamic Banking},
  author       = {Zafar, Muhammad Bilal},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/bilalzafar/IslamicBank-HCDi-RoBERTa}}
}
```
