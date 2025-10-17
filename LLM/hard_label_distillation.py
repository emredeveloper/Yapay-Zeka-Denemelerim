"""
Hard-Label Distillation Implementation
Uses only the final output token from teacher (one-hot encoding)
More memory efficient - similar to DeepSeek-R1 approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os

class HardLabelDistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.temperature = temperature
        
        # Move teacher to same device as student
        if self.args.device is not None:
            self.teacher_model.to(self.args.device)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute distillation loss using teacher's hard predictions
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        # Get teacher's hard predictions (greedy decoding)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits
            # Get argmax as hard labels
            teacher_predictions = teacher_logits.argmax(dim=-1)
        
        # Student forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = teacher_predictions[..., 1:].contiguous()
        
        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Apply temperature scaling and compute cross-entropy loss
        loss = F.cross_entropy(
            shift_logits / self.temperature,
            shift_labels,
            ignore_index=-100
        )
        
        return (loss, outputs) if return_outputs else loss


def train_hard_label_distillation():
    """
    Train a student model using hard-label distillation
    Memory efficient approach used by DeepSeek
    """
    print("=" * 80)
    print("HARD-LABEL DISTILLATION (DeepSeek-R1 Style)")
    print("=" * 80)
    
    # Configuration
    teacher_model_name = "gpt2-medium"  # 355M parameters
    student_model_name = "sshleifer/tiny-gpt2"
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    max_length = 128
    temperature = 1.5
    
    print(f"\nTeacher Model: {teacher_model_name}")
    print(f"Student Model: {student_model_name}")
    print(f"Temperature: {temperature}")
    print(f"Approach: Hard labels only (memory efficient)")
    
    # Load tokenizer and models
    print("\nLoading models...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
    student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
    
    print(f"Teacher parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")
    print(f"Student parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    
    # Calculate memory savings
    vocab_size = student_model.config.vocab_size
    print(f"\nVocab size: {vocab_size:,}")
    print(f"Memory saving: Soft labels would require {vocab_size}x more storage")
    print(f"Hard labels: 1 token ID per position vs {vocab_size} probabilities")
    
    # Load and preprocess dataset
    print("\nLoading dataset...")
    dataset = load_dataset(dataset_name, dataset_config)
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return tokenized
    
    # Use smaller subset for testing
    train_dataset = dataset["train"].select(range(min(1000, len(dataset["train"]))))
    eval_dataset = dataset["validation"].select(range(min(100, len(dataset["validation"]))))
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results_hard_label",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        learning_rate=5e-5,
        logging_dir="./logs_hard_label",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="tensorboard",
    )
    
    # Create custom trainer
    trainer = HardLabelDistillationTrainer(
        teacher_model=teacher_model,
        temperature=temperature,
        model=student_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
    )
    
    # Evaluate before training
    print("\n" + "=" * 80)
    print("EVALUATION BEFORE TRAINING")
    print("=" * 80)
    pre_metrics = trainer.evaluate()
    print(f"Initial Loss: {pre_metrics['eval_loss']:.4f}")
    
    # Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    trainer.train()
    
    # Evaluate after training
    print("\n" + "=" * 80)
    print("EVALUATION AFTER TRAINING")
    print("=" * 80)
    post_metrics = trainer.evaluate()
    print(f"Final Loss: {post_metrics['eval_loss']:.4f}")
    print(f"Improvement: {pre_metrics['eval_loss'] - post_metrics['eval_loss']:.4f}")
    
    # Save the model
    student_model.save_pretrained("./hard_label_student_model")
    tokenizer.save_pretrained("./hard_label_student_model")
    
    # Generate sample text
    print("\n" + "=" * 80)
    print("TEXT GENERATION COMPARISON")
    print("=" * 80)
    
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print(f"\nPrompt: '{prompt}'")
    print("\n--- Teacher Output ---")
    teacher_output = teacher_model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7)
    print(tokenizer.decode(teacher_output[0], skip_special_tokens=True))
    
    print("\n--- Student Output ---")
    student_output = student_model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7)
    print(tokenizer.decode(student_output[0], skip_special_tokens=True))
    
    # Create results summary
    results = {
        "method": "Hard-Label Distillation",
        "teacher_model": teacher_model_name,
        "student_model": student_model_name,
        "teacher_params": sum(p.numel() for p in teacher_model.parameters()),
        "student_params": sum(p.numel() for p in student_model.parameters()),
        "initial_loss": pre_metrics['eval_loss'],
        "final_loss": post_metrics['eval_loss'],
        "improvement": pre_metrics['eval_loss'] - post_metrics['eval_loss'],
        "temperature": temperature,
        "memory_efficient": True,
    }
    
    return results


if __name__ == "__main__":
    results = train_hard_label_distillation()
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    for key, value in results.items():
        print(f"{key}: {value}")
