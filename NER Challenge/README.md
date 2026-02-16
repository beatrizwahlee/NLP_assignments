# Named Entity Recognition for Historical Swedish Text

A machine learning project for identifying named entities in historical Swedish documents using transformer-based models. This system extracts entities such as persons, locations, organizations, dates, and other domain-specific entities from 18th-19th century Swedish text.

## Project Overview

This project implements a Named Entity Recognition (NER) system using fine-tuned transformer models (BERT-based) to identify and classify entities in historical Swedish court records and documents. The model recognizes multiple entity types including persons (PER), locations (LOC), occupations (OCC), organizations (ORG), temporal expressions (TME), and various other categories.

## Entity Types

The model identifies the following entity categories:

- **PER** - Person names
- **LOC** - Location names
- **OCC** - Occupations
- **ORG-INST** - Organizations and Institutions
- **TME-DATE** - Dates
- **TME-TIME** - Times
- **TME-INTRV** - Time intervals
- **WRK** - Works (publications, documents)
- **EVN** - Events
- **SYMP** - Symptoms (medical context)
- **MSR-OTH** - Measurements and other quantifiable information

## Dataset

### Training Data
- **Size**: 677 documents
- **Format**: CSV with columns: `text`, `id`, `entities`
- **Entity Annotations**: Character-level span annotations with start/end positions and labels

### Test Data
- **Size**: 170 documents
- **Format**: CSV with columns: `text`, `id`
- **Output**: Predictions in JSON format with entity spans

## Model Architecture

The project uses transformer-based models for token classification:

1. **Base Model**: Pre-trained Swedish BERT models (e.g., KB-BERT, multilingual BERT)
2. **Task**: Token Classification with BIO (Begin-Inside-Outside) tagging scheme
3. **Fine-tuning**: Custom training on historical Swedish text corpus
4. **Optimization**: Memory-efficient training with gradient accumulation and mixed precision

### Training Configuration

- **Learning Rate**: 2e-5 with linear warmup
- **Batch Size**: 4 (with gradient accumulation)
- **Epochs**: 3-5 epochs
- **Optimizer**: AdamW
- **Loss Function**: Cross-entropy loss with class weights for imbalanced data
- **Evaluation**: F1 score, precision, and recall per entity type

## Technical Features

### Data Preprocessing
- Custom tokenization handling for historical Swedish text
- Alignment of word-level entities with subword tokens
- Special handling of whitespace and punctuation
- BIO tag conversion and label mapping

### Model Enhancements
- CUDA memory optimization with expandable segments
- Custom data collator for token classification
- Weighted loss for handling class imbalance
- Post-processing to merge consecutive B- and I- tags

### Evaluation Metrics
- Per-entity type F1 scores
- Overall precision, recall, and F1
- Confusion matrix analysis
- Entity-level evaluation (exact match)

## Requirements

```
transformers
datasets
torch
nervaluate
seqeval
scikit-learn
pandas
numpy
tqdm
evaluate
```

## Installation

```bash
pip install transformers datasets torch nervaluate seqeval scikit-learn pandas tqdm evaluate
```

## Usage

### Training the Model

```python
# Load and preprocess data
train_df = pd.read_csv("train.csv")

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("model_name")
model = AutoModelForTokenClassification.from_pretrained("model_name", num_labels=num_labels)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### Making Predictions

```python
# Load test data
test_df = pd.read_csv("test.csv")

# Generate predictions
predictions = trainer.predict(test_dataset)

# Post-process and save results
results_df.to_csv("test_predictions.csv", index=False)
```

## File Structure

```
.
├── Code.ipynb                  # Main Jupyter notebook with complete pipeline
├── train.csv                   # Training data with entity annotations
├── test.csv                    # Test data for predictions
├── test_predictions.csv        # Model predictions on test set
└── README.md                   # This file
```

## Output Format

Predictions are saved in CSV format with the following structure:

```csv
id,entities
72774,"[{\"label\": \"PER\", \"start\": 10, \"end\": 25}, ...]"
```

Each entity is represented as a JSON object with:
- `label`: Entity type (e.g., PER, LOC, TME-DATE)
- `start`: Character position where entity begins
- `end`: Character position where entity ends

## Performance Considerations

### Memory Optimization
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation
- Use gradient accumulation for larger effective batch sizes
- Employ mixed precision training (FP16) when available

### Training Time
- GPU recommended (CUDA-compatible)
- Typical training time: 1-3 hours on modern GPU (depends on model size)
- CPU training possible but significantly slower

## Model Evaluation

The model is evaluated using:
- **Strict evaluation**: Exact boundary and label match
- **Lenient evaluation**: Overlapping spans with correct label
- **Per-entity metrics**: Individual F1 scores for each entity type
- **Macro/Micro averaging**: Overall performance metrics

## Challenges and Solutions

### Historical Text Challenges
- **Spelling variations**: Historical Swedish has inconsistent spelling
- **Solution**: Use robust pre-trained models and augmentation

### Class Imbalance
- **Issue**: Some entity types are rare in the corpus
- **Solution**: Weighted loss function and oversampling rare classes

### Long Documents
- **Issue**: Historical documents can exceed model's max sequence length
- **Solution**: Sliding window approach with overlapping segments

## Future Improvements

- [ ] Experiment with larger Swedish language models
- [ ] Implement active learning for better annotation
- [ ] Add entity linking to knowledge bases
- [ ] Develop ensemble methods for improved accuracy
- [ ] Create a web interface for interactive predictions

## Contributing

Contributions are welcome! Areas for contribution:
- Additional pre-trained models testing
- Improved post-processing heuristics
- Extended entity type support
- Performance optimization

## Citation

If you use this code in your research, please cite:

```
[Add citation information here]
```

## License

[Add license information here]

## Contact

[Add contact information here]

## Acknowledgments

- Hugging Face Transformers library
- Swedish Language Models (KB-BERT, etc.)
- Historical document providers
- Open-source NLP community
