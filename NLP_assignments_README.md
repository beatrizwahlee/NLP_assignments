# NLP Assignments Collection

A collection of Natural Language Processing projects focusing on transformer-based models for real-world text analysis tasks. This repository contains two comprehensive deep learning challenges demonstrating advanced NLP techniques.

## 📚 Overview

This repository showcases two distinct NLP challenges solved using state-of-the-art transformer architectures (BERT-based models). Each project addresses different aspects of text understanding: entity recognition in historical documents and multi-label classification of clinical texts.

## 🎯 Projects

### 1. Named Entity Recognition (NER) - Historical Swedish Text

**Challenge**: Extract and classify named entities from 18th-19th century Swedish court records and historical documents.

**Key Highlights**:
- **Domain**: Historical text analysis
- **Language**: Swedish (historical orthography)
- **Task**: Token classification with 12 entity types
- **Architecture**: Fine-tuned BERT for sequence labeling
- **Dataset**: 677 training documents, 170 test documents
- **Entity Types**: Persons, locations, organizations, dates, times, occupations, symptoms, events, and more

**Technical Features**:
- BIO tagging scheme for entity boundaries
- Custom handling of historical text variations
- Character-level span annotations
- Weighted loss for class imbalance
- Memory-optimized training pipeline

**Performance**: F1-score evaluation with per-entity metrics and confusion matrix analysis

**📁 Location**: `./NER_challenge/`

---

### 2. Multi-Label Clinical Text Classification

**Challenge**: Predict multiple medical conditions from Spanish-language clinical case reports.

**Key Highlights**:
- **Domain**: Clinical/Medical text analysis
- **Language**: Spanish
- **Task**: Multi-label classification
- **Architecture**: BERT with custom classifier head
- **Dataset**: 7,543 training samples
- **Labels**: Multiple concurrent medical conditions per case

**Technical Features**:
- Multi-label learning (multiple positive labels per sample)
- Text data augmentation techniques
- Binary cross-entropy loss optimization
- Macro F1-score evaluation for balanced performance
- Custom PyTorch dataset implementation

**Performance**: Best macro F1-score of 0.9138

**📁 Location**: `./Clinical_Text_Classification/`

---

## 🔬 Comparison of Challenges

| Aspect | NER Challenge | Clinical Classification |
|--------|---------------|------------------------|
| **Task Type** | Sequence Labeling | Document Classification |
| **Output** | Entity spans + labels | Multi-label predictions |
| **Language** | Swedish (historical) | Spanish (modern) |
| **Domain** | Legal/Historical | Medical/Clinical |
| **Model Output** | Token-level tags | Document-level labels |
| **Main Challenge** | Historical spelling variations | Multi-label dependencies |
| **Evaluation** | Per-entity F1 scores | Macro F1-score |

## 🛠️ Common Technical Stack

Both projects leverage:

- **Framework**: PyTorch
- **Transformers**: Hugging Face Transformers library
- **Models**: BERT-based architectures
- **Hardware**: CUDA-enabled GPU training
- **Languages**: Python 3.7+
- **Tools**: Jupyter notebooks for experimentation

### Shared Dependencies

```bash
pip install transformers datasets torch scikit-learn pandas numpy tqdm
```

## 📊 Dataset Characteristics

### NER Dataset
- **Format**: CSV with text and JSON entity annotations
- **Annotation Style**: Character-level spans with entity types
- **Challenge**: Inconsistent historical spelling and terminology
- **Size**: 677 training + 170 test documents

### Clinical Classification Dataset
- **Format**: CSV with text and multi-label arrays
- **Annotation Style**: Binary vectors for condition presence
- **Challenge**: Label imbalance and co-occurrence patterns
- **Size**: 7,543 training samples

## 🚀 Getting Started

### Prerequisites

```bash
# Clone the repository
git clone [repository-url]
cd NLP_assignments

# Install common dependencies
pip install transformers datasets torch scikit-learn pandas tqdm nltk
```

### Running the Projects

**For NER Challenge:**
```bash
cd NER_challenge
jupyter notebook Code.ipynb
```

**For Clinical Classification:**
```bash
cd Clinical_Text_Classification
jupyter notebook Code.ipynb
```

## 🎓 Learning Objectives

These projects demonstrate proficiency in:

1. **Transformer Architecture**: Fine-tuning pre-trained BERT models
2. **Task-Specific Adaptations**: Sequence labeling vs. document classification
3. **Multi-lingual NLP**: Working with Swedish and Spanish texts
4. **Domain Expertise**: Historical and medical text processing
5. **Data Handling**: Custom datasets, augmentation, and preprocessing
6. **Evaluation Metrics**: Task-appropriate metric selection
7. **Production Considerations**: Memory optimization and inference efficiency

## 💡 Key Techniques Demonstrated

### Natural Language Processing
- Tokenization and subword handling
- Attention mechanisms
- Transfer learning with pre-trained models
- Fine-tuning strategies

### Machine Learning
- Multi-label classification
- Sequence labeling with BIO tagging
- Class imbalance handling
- Loss function selection
- Hyperparameter optimization

### Software Engineering
- Modular code organization
- Custom PyTorch datasets
- Memory-efficient training
- Reproducible experiments with seed setting
- Progress tracking and logging

## 📈 Results Summary

### NER Challenge
- Successfully identifies 12 entity types in historical Swedish text
- Handles spelling variations and archaic language structures
- Produces character-level entity annotations
- Evaluated with strict and lenient matching criteria

### Clinical Classification
- Achieves 91.38% macro F1-score on validation set
- Handles multiple concurrent conditions per case
- Maintains balanced performance across label classes
- Processes Spanish medical terminology effectively

## 🔮 Potential Extensions

### For NER Project
- [ ] Entity linking to historical knowledge bases
- [ ] Temporal relationship extraction
- [ ] Cross-lingual transfer to other Scandinavian languages
- [ ] Active learning for annotation efficiency

### For Clinical Classification
- [ ] Multi-task learning with auxiliary objectives
- [ ] Explainability with attention visualization
- [ ] Integration with medical ontologies (ICD codes)
- [ ] Few-shot learning for rare conditions

## 🏆 Applications

### NER System
- Digital humanities research
- Historical document digitization
- Genealogical research tools
- Legal document analysis

### Clinical Classifier
- Clinical decision support systems
- Medical record analysis
- Patient stratification
- Literature review automation

## 📝 Documentation

Each project contains its own detailed README with:
- Complete setup instructions
- Architecture explanations
- Training procedures
- Evaluation metrics
- Usage examples
- Performance benchmarks

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional pre-trained model experiments
- Enhanced data augmentation strategies
- Performance optimization
- Extended evaluation metrics
- Documentation improvements

## 📚 References

### Papers & Resources
- BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)
- Named Entity Recognition in Historical Texts
- Multi-Label Text Classification with Transformers
- Clinical NLP Best Practices

### Libraries & Tools
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)

## 📜 License

[Specify your license here]

## 👥 Author

[Your name/information here]

## 📧 Contact

For questions or collaboration opportunities:
- [Your contact information]
- [GitHub profile]
- [Email]

---

**Note**: These projects were developed as part of NLP coursework/challenges and demonstrate practical applications of transformer-based models in specialized domains. Training was conducted using Google Colab with T4 GPU acceleration.

## 🌟 Acknowledgments

- Hugging Face for the transformers library
- PyTorch team for the deep learning framework
- Dataset providers and annotators
- Open-source NLP community
- Course instructors and mentors
