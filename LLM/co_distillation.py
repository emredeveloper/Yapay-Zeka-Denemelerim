"""
Co-Distillation Implementation
Both teacher and student train simultaneously
Teacher learns from hard labels, student learns from teacher's soft predictions
Similar to Llama 4 approach (Behemoth -> Scout & Maverick)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os

class CoDistillationTrainer:
    def __init__(
        self,
        teacher_model,
        student_model,
        tokenizer,
        train_dataset,
        eval_dataset,
        output_dir="./results_co_distillation",
        num_epochs=3,
        batch_size=4,
        learning_rate_teacher=5e-5,
        learning_rate_student=5e-5,
        temperature=2.0,
        device=None
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.temperature = temperature
        
        # Setup device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_model.to(self.device)
        self.student_model.to(self.device)
        
        # Setup optimizers
        self.teacher_optimizer = torch.optim.AdamW(
            self.teacher_model.parameters(),
            lr=learning_rate_teacher
        )
        self.student_optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate_student
        )
        
        os.makedirs(output_dir, exist_ok=True)
    
    def compute_teacher_loss(self, logits, labels):
        """
        Standard cross-entropy loss for teacher on hard labels
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        return loss
    
    def compute_student_loss(self, student_logits, teacher_logits, labels):
        """
        Distillation loss for student combining soft and hard targets
        """
        # Shift for next-token prediction
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        shift_student_logits = shift_student_logits.view(-1, shift_student_logits.size(-1))
        shift_teacher_logits = shift_teacher_logits.view(-1, shift_teacher_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Hard loss
        hard_loss = F.cross_entropy(shift_student_logits, shift_labels, ignore_index=-100)
        
        # Soft loss (KL divergence)
        soft_student = F.log_softmax(shift_student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(shift_teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        # Combine losses
        total_loss = 0.5 * hard_loss + 0.5 * soft_loss
        return total_loss, hard_loss, soft_loss
    
    def train_step(self, batch):
        """
        Single training step for both models
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = input_ids.clone()
        
        # Teacher forward pass and update
        self.teacher_model.train()
        teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        teacher_logits = teacher_outputs.logits
        teacher_loss = self.compute_teacher_loss(teacher_logits, labels)
        
        self.teacher_optimizer.zero_grad()
        teacher_loss.backward()
        self.teacher_optimizer.step()
        
        # Student forward pass (use updated teacher logits)
        self.student_model.train()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits
        
        student_outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits
        student_loss, hard_loss, soft_loss = self.compute_student_loss(
            student_logits, teacher_logits, labels
        )
        
        self.student_optimizer.zero_grad()
        student_loss.backward()
        self.student_optimizer.step()
        
        return {
            "teacher_loss": teacher_loss.item(),
            "student_loss": student_loss.item(),
            "student_hard_loss": hard_loss.item(),
            "student_soft_loss": soft_loss.item(),
        }
    
    def evaluate(self):
        """
        Evaluate both models
        """
        self.teacher_model.eval()
        self.student_model.eval()
        
        total_teacher_loss = 0
        total_student_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(self.eval_dataset), self.batch_size):
                batch_data = self.eval_dataset[i:min(i+self.batch_size, len(self.eval_dataset))]
                
                input_ids = torch.tensor(batch_data["input_ids"]).to(self.device)
                attention_mask = torch.tensor(batch_data["attention_mask"]).to(self.device)
                labels = input_ids.clone()
                
                # Teacher evaluation
                teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                teacher_loss = self.compute_teacher_loss(teacher_outputs.logits, labels)
                
                # Student evaluation
                student_outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask)
                student_loss, _, _ = self.compute_student_loss(
                    student_outputs.logits, teacher_outputs.logits, labels
                )
                
                total_teacher_loss += teacher_loss.item()
                total_student_loss += student_loss.item()
                num_batches += 1
        
        return {
            "teacher_loss": total_teacher_loss / num_batches,
            "student_loss": total_student_loss / num_batches,
        }
    
    def train(self):
        """
        Main training loop
        """
        print("\n" + "=" * 80)
        print("STARTING CO-DISTILLATION TRAINING")
        print("=" * 80)
        
        # Initial evaluation
        print("\nInitial Evaluation:")
        initial_metrics = self.evaluate()
        print(f"  Teacher Loss: {initial_metrics['teacher_loss']:.4f}")
        print(f"  Student Loss: {initial_metrics['student_loss']:.4f}")
        
        # Training loop
        for epoch in range(self.num_epochs):
            print(f"\n{'=' * 80}")
            print(f"EPOCH {epoch + 1}/{self.num_epochs}")
            print("=" * 80)
            
            epoch_metrics = {
                "teacher_loss": 0,
                "student_loss": 0,
                "student_hard_loss": 0,
                "student_soft_loss": 0,
            }
            num_batches = 0
            
            # Training batches
            for i in tqdm(range(0, len(self.train_dataset), self.batch_size), desc=f"Epoch {epoch+1}"):
                batch_data = self.train_dataset[i:min(i+self.batch_size, len(self.train_dataset))]
                
                batch = {
                    "input_ids": torch.tensor(batch_data["input_ids"]),
                    "attention_mask": torch.tensor(batch_data["attention_mask"]),
                }
                
                metrics = self.train_step(batch)
                
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
                num_batches += 1
            
            # Average metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
            
            print(f"\nEpoch {epoch + 1} Training Metrics:")
            print(f"  Teacher Loss: {epoch_metrics['teacher_loss']:.4f}")
            print(f"  Student Total Loss: {epoch_metrics['student_loss']:.4f}")
            print(f"  Student Hard Loss: {epoch_metrics['student_hard_loss']:.4f}")
            print(f"  Student Soft Loss: {epoch_metrics['student_soft_loss']:.4f}")
            
            # Evaluation
            eval_metrics = self.evaluate()
            print(f"\nEpoch {epoch + 1} Evaluation:")
            print(f"  Teacher Loss: {eval_metrics['teacher_loss']:.4f}")
            print(f"  Student Loss: {eval_metrics['student_loss']:.4f}")
        
        # Final evaluation
        print("\n" + "=" * 80)
        print("FINAL EVALUATION")
        print("=" * 80)
        final_metrics = self.evaluate()
        print(f"Teacher Loss: {final_metrics['teacher_loss']:.4f}")
        print(f"Student Loss: {final_metrics['student_loss']:.4f}")
        print(f"\nTeacher Improvement: {initial_metrics['teacher_loss'] - final_metrics['teacher_loss']:.4f}")
        print(f"Student Improvement: {initial_metrics['student_loss'] - final_metrics['student_loss']:.4f}")
        
        return initial_metrics, final_metrics


def train_co_distillation():
    """
    Train both teacher and student models simultaneously
    """
    print("=" * 80)
    print("CO-DISTILLATION (Llama 4 Style)")
    print("=" * 80)
    
    # Configuration
    teacher_model_name = "gpt2-medium"  # 355M parameters - larger teacher
    student_model_name = "sshleifer/tiny-gpt2"  # 11M parameters - smaller student
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    max_length = 128
    temperature = 2.0
    
    print(f"\nTeacher Model: {teacher_model_name}")
    print(f"Student Model: {student_model_name}")
    print(f"Temperature: {temperature}")
    print(f"Approach: Simultaneous training (teacher + student)")
    
    # Load tokenizer and models
    print("\nLoading models...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load two separate instances
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
    student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
    
    print(f"Teacher parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")
    print(f"Student parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    
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
    
    # Create trainer
    trainer = CoDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        num_epochs=3,
        batch_size=4,
        temperature=temperature,
    )
    
    # Train
    initial_metrics, final_metrics = trainer.train()
    
    # Save models
    teacher_model.save_pretrained("./co_distillation_teacher_model")
    student_model.save_pretrained("./co_distillation_student_model")
    tokenizer.save_pretrained("./co_distillation_teacher_model")
    tokenizer.save_pretrained("./co_distillation_student_model")
    
    # Generate sample text
    print("\n" + "=" * 80)
    print("TEXT GENERATION COMPARISON")
    print("=" * 80)
    
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(trainer.device)
    
    print(f"\nPrompt: '{prompt}'")
    print("\n--- Teacher Output ---")
    teacher_output = teacher_model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7)
    print(tokenizer.decode(teacher_output[0], skip_special_tokens=True))
    
    print("\n--- Student Output ---")
    student_output = student_model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7)
    print(tokenizer.decode(student_output[0], skip_special_tokens=True))
    
    # Create results summary
    results = {
        "method": "Co-Distillation",
        "teacher_model": teacher_model_name,
        "student_model": student_model_name,
        "teacher_params": sum(p.numel() for p in teacher_model.parameters()),
        "student_params": sum(p.numel() for p in student_model.parameters()),
        "initial_teacher_loss": initial_metrics['teacher_loss'],
        "initial_student_loss": initial_metrics['student_loss'],
        "final_teacher_loss": final_metrics['teacher_loss'],
        "final_student_loss": final_metrics['student_loss'],
        "teacher_improvement": initial_metrics['teacher_loss'] - final_metrics['teacher_loss'],
        "student_improvement": initial_metrics['student_loss'] - final_metrics['student_loss'],
        "temperature": temperature,
    }
    
    return results


if __name__ == "__main__":
    results = train_co_distillation()
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    for key, value in results.items():
        print(f"{key}: {value}")
