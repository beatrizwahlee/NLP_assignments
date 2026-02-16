# Clinical Text Multi-Label Classification

A deep learning project for multi-label classification of clinical case reports in Spanish using transformer-based models (BERT). This project predicts multiple medical conditions from clinical text descriptions.

## 📋 Project Overview

This project implements a multi-label text classification system designed to analyze clinical case reports and identify the presence of various medical conditions. The model processes Spanish-language clinical narratives and outputs predictions for multiple diagnostic labels simultaneously.

## 🎯 Key Features

- **Multi-label Classification**: Predicts multiple conditions per clinical case
- **Transformer Architecture**: Utilizes BERT-based models for state-of-the-art NLP performance
- **Data Augmentation**: Implements text augmentation techniques to improve model robustness
- **Custom Dataset Handling**: Specialized PyTorch dataset class for clinical text
- **Comprehensive Evaluation**: Macro F1-score tracking for balanced performance across classes
- **GPU Acceleration**: Optimized for CUDA-enabled training

## 🗂️ Project Structure

```
.
├── Code.ipynb              # Main Jupyter notebook with complete pipeline
├── train.csv               # Training dataset with clinical texts and labels
├── test.csv                # Test dataset for predictions
├── test_prediction.csv     # Model predictions on test set
└── README.md              # Project documentation
```

## 📊 Dataset

### Training Data
- **Format**: CSV file with columns: `id`, `text`, `labels`
- **Text**: Clinical case descriptions in Spanish
- **Labels**: Multi-label format `[0, 0, 0, 0]` representing different medical conditions
- **Size**: 7,543 training samples

### Test Data
- Clinical texts without labels for prediction
- Same format as training data minus the labels column

## 🛠️ Technical Stack

### Dependencies

```python
transformers        # Hugging Face transformers library
datasets           # Dataset handling and processing
torch              # PyTorch deep learning framework
scikit-learn       # Machine learning utilities
pandas             # Data manipulation
tqdm               # Progress bars
nltk               # Natural language processing
nlpaug             # Text data augmentation
```

### Model Architecture

- **Base Model**: Pre-trained BERT transformer
- **Custom Classifier**: Linear layer for multi-label output
- **Loss Function**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Linear learning rate decay with warmup

## 🚀 Getting Started

### Installation

```bash
pip install transformers datasets torch scikit-learn pandas tqdm nltk nlpaug
```

### Hardware Requirements

- **GPU**: CUDA-compatible GPU (T4 or better recommended)
- **RAM**: Minimum 16GB recommended
- **Storage**: At least 5GB for model and data

## 📈 Training Pipeline

1. **Data Loading & Preprocessing**
   - Load CSV files with pandas
   - Parse multi-label format
   - Text cleaning and normalization

2. **Data Augmentation** (Optional)
   - Synonym replacement
   - Random insertion/deletion
   - Back-translation techniques

3. **Tokenization**
   - BERT tokenizer with max length padding
   - Attention mask generation
   - Special token handling

4. **Model Training**
   - 10 epochs with early stopping potential
   - Batch size: 8
   - Learning rate scheduling
   - Gradient accumulation

5. **Evaluation**
   - Macro F1-score calculation
   - Per-class performance metrics
   - Best model checkpointing

## 🎓 Model Training Details

### Hyperparameters

```python
Epochs: 10
Batch Size: 8
Learning Rate: 2e-5 (with warmup and decay)
Optimizer: AdamW
Loss Function: BCEWithLogitsLoss
Evaluation Metric: Macro F1-Score
```

### Training Results

- **Final Training Loss**: 0.001024
- **Best Macro F1-Score**: 0.9138
- **Training Time**: ~40-50 seconds per epoch on T4 GPU

## 🔮 Inference

The model generates predictions with:
- **Threshold**: 0.5 for binary classification
- **Output Format**: Multi-label array per sample
- **Submission File**: CSV with `id`, `text`, and formatted predictions

## 📤 Output Format

Predictions are saved in the following format:

```csv
id,text,pred
test_0,"Clinical text...","[0, 1, 0, 1]"
test_1,"Clinical text...","[1, 0, 0, 0]"
```

## 🔍 Key Components

### Custom Dataset Class
```python
class ClinicalDataset(Dataset):
    - Handles tokenization
    - Manages attention masks
    - Processes multi-label targets
```

### Model Architecture
```python
class ClinicalBERTModel(nn.Module):
    - BERT encoder
    - Dropout layer
    - Linear classifier
```

### Training Loop
- Batch processing with progress tracking
- Loss calculation and backpropagation
- Model checkpoint saving
- Validation with F1-score

## 📊 Performance Metrics

The model is evaluated using:
- **Macro F1-Score**: Equal weight to all classes
- **Training Loss**: Binary cross-entropy
- **Per-epoch Evaluation**: Consistent monitoring

## 🔧 Customization

To adapt this project for other datasets:

1. Update the number of labels in model initialization
2. Modify tokenizer for different languages if needed
3. Adjust batch size based on available GPU memory
4. Tune hyperparameters (learning rate, epochs, etc.)

## ⚠️ Important Notes

- Text data is in Spanish - ensure appropriate language model
- Multi-label classification allows multiple positive labels per sample
- Model checkpoints save only the best performing version
- GPU acceleration is highly recommended for training

## 🐛 Troubleshooting

**Out of Memory Errors**
- Reduce batch size
- Decrease maximum sequence length
- Use gradient accumulation

**Poor Performance**
- Increase training epochs
- Adjust learning rate
- Add more data augmentation
- Try different pre-trained models

## 📝 Citation

If you use this code, please acknowledge the use of:
- Hugging Face Transformers library
- PyTorch framework
- Original clinical dataset sources (if applicable)

## 📜 License

This project is provided as-is for educational and research purposes.

## 👥 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:
- Bug fixes
- Performance improvements
- Additional features
- Documentation enhancements

## 🔗 Resources

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Multi-label Classification Guide](https://scikit-learn.org/stable/modules/multiclass.html)

---

**Note**: This project was developed using Google Colab with T4 GPU acceleration. Training times and memory requirements may vary based on your hardware configuration.
