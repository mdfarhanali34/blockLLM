"""
BlockLLM: Memory-efficient training for large language models.
This script trains a model on IMDB with <5% of parameters updated, reducing memory usage.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from blockllm_torch.blockllm import BlockLLM, BlockLLMConfig
from torch.utils.data import DataLoader
from datasets import load_dataset

# Reduce max_length to 128 to prevent position embedding overflow
def prepare_data(tokenizer, max_length=128):
    dataset = load_dataset("HuggingFaceH4/MATH-500")

    def tokenize_function(examples):
        texts = [
            f"Review: {text}\nSentiment: {'positive' if label == 1 else 'negative'}"
            for text, label in zip(examples['text'], examples['label'])
        ]
        texts = [t for t in texts if t.strip()]
        if not texts:
            return {'input_ids': [], 'attention_mask': [], 'labels': []}

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )

        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        batch_size=100
    )

    tokenized_dataset = tokenized_dataset.filter(
        lambda example: len(example['input_ids']) > 0
    )

    return tokenized_dataset

# Fix collate function to handle padding properly
def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    labels = torch.stack([torch.tensor(item['labels']) for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def train(model, tokenizer, train_dataloader, optimizer, num_epochs=3, device='cuda', scheduler=None, gradient_accumulation_steps=4):
    model.train()
    print(f"Training on device {device}")

    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps  # Normalize loss for accumulation

            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()

            if scheduler:
                scheduler.step()

            total_loss += loss.item() * gradient_accumulation_steps  # Re-adjust loss for logging

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item() * gradient_accumulation_steps:.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Sample text generation every 2 epochs
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                sample_text = "Review: This movie was absolutely"
                inputs = tokenizer(sample_text, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"\nSample generation:\n{generated_text}\n")
            model.train()

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 10
    torch.manual_seed(42)

    model_name = "distilgpt2"  # Small GPT-2 variant for efficiency
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure correct padding settings for GPT-2
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    tokenized_dataset = prepare_data(tokenizer)

    train_dataloader = DataLoader(
        tokenized_dataset['train'],
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Configure BlockLLM optimizer with <5% updates
    config = BlockLLMConfig(
        lr=3e-5,  
        sparsity_level=0.95,  # 95% parameters frozen, only 5% updated
        update_freq=2000,  # Reduce update frequency
        num_bottom_to_sample=3,  
        patience=200,  
        param_update_interval=1000  
    )

    optimizer = BlockLLM(
        model.named_parameters(),
        config=config
    )

    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    train(model, tokenizer, train_dataloader, optimizer, num_epochs=num_epochs, device=device, scheduler=scheduler)

if __name__ == "__main__":
    main()
