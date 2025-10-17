# LLM Distillation Techniques Implementation

This project implements three different knowledge distillation techniques for Large Language Models using Hugging Face Transformers.

## Overview

Knowledge distillation is a technique to transfer knowledge from a larger "teacher" model to a smaller "student" model, creating more efficient models while maintaining performance.

## Implemented Methods

### 1. Soft-Label Distillation
- **Description**: Uses full probability distribution from teacher model
- **Pros**: Maximum knowledge transfer, captures teacher's uncertainty
- **Cons**: Memory intensive (vocab_size × num_tokens storage)
- **File**: `soft_label_distillation.py`

**Memory Requirements Example:**
- Vocab size: 100,000 tokens
- Corpus: 5 trillion tokens
- Storage needed: ~500 million GB (float8)

### 2. Hard-Label Distillation (DeepSeek-R1 Style)
- **Description**: Uses only final output token from teacher (one-hot encoding)
- **Pros**: Extremely memory efficient, scalable to massive datasets
- **Cons**: Less information transfer than soft labels
- **File**: `hard_label_distillation.py`
- **Reference**: DeepSeek-R1 → Qwen & Llama 3.1

**Memory Savings:**
- Stores 1 token ID per position instead of vocab_size probabilities
- ~100,000x more memory efficient than soft labels

### 3. Co-Distillation (Llama 4 Style)
- **Description**: Teacher and student train simultaneously
- **Approach**: 
  - Teacher learns from hard labels (standard training)
  - Student learns from teacher's soft probabilities
  - Both models improve together
- **Pros**: No need for pre-trained teacher, both models benefit
- **Cons**: Requires training both models (more compute)
- **File**: `co_distillation.py`
- **Reference**: Llama 4 Behemoth → Scout & Maverick

## Installation

1. Create and activate Python virtual environment (you already have .venv):
```bash
.venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run All Methods
To run all three distillation techniques and compare results:
```bash
python run_all_distillation.py
```

### Run Individual Methods
```bash
# Soft-label distillation
python soft_label_distillation.py

# Hard-label distillation
python hard_label_distillation.py

# Co-distillation
python co_distillation.py
```

## Models Used

For testing purposes, we use small models:
- **Teacher**: `gpt2-medium` (~355M parameters)
- **Student**: `sshleifer/tiny-gpt2` (~11M parameters)
- **Dataset**: WikiText-2 (small subset for quick testing)

## Output

### Trained Models
Each method saves trained models to:
- `./soft_label_student_model/`
- `./hard_label_student_model/`
- `./co_distillation_teacher_model/`
- `./co_distillation_student_model/`

### Results
- JSON file with detailed metrics: `distillation_results_YYYYMMDD_HHMMSS.json`
- TensorBoard logs in respective `./logs_*` directories
- Console output with training progress and comparisons

### Metrics Tracked
- Initial and final loss values
- Loss improvement
- Training time
- Model parameters
- Text generation samples

## Viewing Training Progress

TensorBoard logs are saved for each method:
```bash
# View soft-label training
tensorboard --logdir=./logs_soft_label

# View hard-label training
tensorboard --logdir=./logs_hard_label
```

## Key Findings

### When to Use Each Method

**Soft-Label Distillation:**
- You have access to teacher weights
- Memory is not a constraint
- Want maximum knowledge transfer
- Working with smaller datasets

**Hard-Label Distillation:**
- Large-scale deployment (trillion+ tokens)
- Memory constrained
- Only have access to teacher's outputs
- Need scalability over maximum transfer

**Co-Distillation:**
- Training from scratch
- Want both models to benefit
- Have computational resources for dual training
- Building model families (like Llama 4)

## Technical Details

### Temperature Scaling
Temperature parameter controls the "softness" of probability distributions:
- Higher temperature (2-4): Softer distributions, more knowledge transfer
- Lower temperature (1): Sharper distributions, closer to hard labels

### Loss Functions

**Soft-Label:**
```python
loss = α × CE(student, labels) + (1-α) × KL(student || teacher) × T²
```

**Hard-Label:**
```python
loss = CE(student, argmax(teacher)) / T
```

**Co-Distillation:**
```python
teacher_loss = CE(teacher, labels)
student_loss = 0.5 × CE(student, labels) + 0.5 × KL(student || teacher) × T²
```

## References

1. **Hinton et al. (2015)** - "Distilling the Knowledge in a Neural Network"
2. **DeepSeek-R1** - Hard-label distillation for large-scale LLMs
3. **Llama 4** - Co-distillation for model families (Behemoth → Scout & Maverick)

## Customization

To use your own models and datasets, modify the configuration in each script:
```python
teacher_model_name = "your-teacher-model"
student_model_name = "your-student-model"
dataset_name = "your-dataset"
```

## Performance Tips

1. **GPU Usage**: Training will automatically use CUDA if available
2. **Batch Size**: Adjust `per_device_train_batch_size` based on GPU memory
3. **Dataset Size**: Start with small subsets for testing, scale up for production
4. **Mixed Precision**: Add `fp16=True` to TrainingArguments for faster training

## Troubleshooting

**Out of Memory:**
- Reduce batch size
- Use smaller max_length
- Use hard-label distillation instead of soft-label
- Enable gradient checkpointing

**Slow Training:**
- Reduce dataset size
- Increase batch size (if memory allows)
- Use multiple GPUs with accelerate

## License

MIT License - Feel free to use and modify for your projects.

## Citation

If you use this implementation in your research, please cite the original papers for each method.
