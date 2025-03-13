"""
Memory-efficient LLM training comparison: BlockLLM vs GaLore

This script:
1. Implements both BlockLLM and GaLore optimization methods
2. Trains the same model with different optimization techniques
3. Compares memory usage, training speed, and performance
4. Visualizes results using matplotlib
"""

import torch
import torch.nn as nn
import os
import argparse
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from blockllm_torch.blockllm import BlockLLM, BlockLLMConfig
from galore_torch import GaLoreAdamW
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# For memory tracking
try:
    import psutil
    memory_tracking_available = True
except ImportError:
    memory_tracking_available = False
    print("psutil not available, detailed memory tracking will be disabled")

def parse_args():
    parser = argparse.ArgumentParser(description="Compare memory-efficient optimization methods")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Base model to fine-tune")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500", help="Dataset to train on")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--methods", type=str, default="blockllm,galore,full", 
                        help="Comma-separated list of methods to compare (blockllm,galore,full)")
    parser.add_argument("--sparsity", type=float, default=0.95, help="Sparsity level for BlockLLM")
    parser.add_argument("--rank", type=int, default=128, help="Rank for GaLore optimization")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--update_proj_gap", type=int, default=200, help="Projection update frequency for GaLore")
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_memory_usage():
    """Get current memory usage in GB"""
    if memory_tracking_available:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024 ** 3)
    else:
        memory_gb = 0
        
    if torch.cuda.is_available():
        torch_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        torch_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        return {
            "process_memory_gb": memory_gb,
            "torch_allocated_gb": torch_memory,
            "torch_reserved_gb": torch_reserved
        }
    return {"process_memory_gb": memory_gb}

def prepare_data(tokenizer, dataset_name, max_length=128):
    """Prepare dataset for training"""
    print(f"Loading dataset: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to 'imdb' dataset")
        dataset = load_dataset("imdb")
    
    # Determine which splits to use
    available_splits = list(dataset.keys())
    print(f"Available splits: {available_splits}")
    
    train_split = "train" if "train" in available_splits else available_splits[0]
    val_split = "test" if "test" in available_splits else "validation" if "validation" in available_splits else train_split
    
    # Check dataset structure
    example = dataset[train_split][0]
    
    # Determine text field
    if "text" in example:
        text_field = "text"
    elif "review" in example:
        text_field = "review"
    else:
        text_field = next((k for k in example.keys() if isinstance(example[k], str)), None)
    
    if not text_field:
        raise ValueError(f"Could not find text field in dataset with keys: {example.keys()}")
    
    print(f"Using '{text_field}' as the text field")
    
    # Define tokenization function
    def tokenize_function(examples):
        texts = examples[text_field]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized
    
    # Tokenize dataset
    tokenized_train = dataset[train_split].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset[train_split].column_names,
        batch_size=100
    )
    
    tokenized_val = dataset[val_split].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset[val_split].column_names,
        batch_size=100
    )
    
    # Take a subset for faster comparisons
    train_subset_size = min(2000, len(tokenized_train))
    val_subset_size = min(200, len(tokenized_val))
    
    train_indices = np.random.choice(len(tokenized_train), train_subset_size, replace=False)
    val_indices = np.random.choice(len(tokenized_val), val_subset_size, replace=False)
    
    train_dataset = torch.utils.data.Subset(tokenized_train, train_indices)
    val_dataset = torch.utils.data.Subset(tokenized_val, val_indices)
    
    print(f"Using {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    
    return train_dataset, val_dataset

def collate_fn(batch):
    """Collate batch for DataLoader"""
    input_ids = torch.stack([torch.tensor(batch[i]['input_ids']) for i in range(len(batch))])
    attention_mask = torch.stack([torch.tensor(batch[i]['attention_mask']) for i in range(len(batch))])
    labels = torch.stack([torch.tensor(batch[i]['labels']) for i in range(len(batch))])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def create_optimizer(method, model, args):
    """Create optimizer based on method"""
    if method == "blockllm":
        # BlockLLM optimizer
        config = BlockLLMConfig(
            lr=args.lr,
            sparsity_level=args.sparsity,
            update_freq=args.update_proj_gap,
            num_bottom_to_sample=3,
            patience=200,
            param_update_interval=args.update_proj_gap
        )
        optimizer = BlockLLM(model.named_parameters(), config=config)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        active_param_count = int(param_count * (1 - args.sparsity))
        print(f"BlockLLM: Total parameters: {param_count:,}, Active parameters: {active_param_count:,} ({(1-args.sparsity)*100:.1f}%)")
        
    elif method == "galore":
        # GaLore optimizer
        galore_params = []
        non_galore_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.ndim >= 2 and min(param.shape) >= args.rank:
                    galore_params.append(param)
                else:
                    non_galore_params.append(param)
        
        galore_param_count = sum(p.numel() for p in galore_params)
        non_galore_param_count = sum(p.numel() for p in non_galore_params)
        total_param_count = galore_param_count + non_galore_param_count
        
        print(f"GaLore: Total parameters: {total_param_count:,}")
        print(f"  - Parameters using GaLore: {galore_param_count:,} ({galore_param_count/total_param_count*100:.1f}%)")
        print(f"  - Parameters using standard updates: {non_galore_param_count:,} ({non_galore_param_count/total_param_count*100:.1f}%)")
        
        param_groups = [
            {'params': non_galore_params},
            {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': 0.25, 'proj_type': 'std'}
        ]
        optimizer = GaLoreAdamW(param_groups, lr=args.lr)
        
    else:  # "full" standard optimization
        # Standard AdamW optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full Fine-tuning: Total parameters: {param_count:,}")
    
    return optimizer

def evaluate(model, dataloader, device):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train_model(model, method, train_dataloader, val_dataloader, args, device):
    """Train model with specified optimization method"""
    print(f"\n{'='*50}")
    print(f"Training with {method} optimization")
    print(f"{'='*50}")
    
    # Create fresh model copy to ensure fair comparison
    model_copy = AutoModelForCausalLM.from_pretrained(args.model_name)
    model_copy.to(device)
    
    # Reset CUDA memory tracking for accurate measurements
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()
    
    # Create optimizer for this method
    optimizer = create_optimizer(method, model_copy, args)
    
    # Create scheduler
    num_training_steps = len(train_dataloader) * args.epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # Initialize tracking variables
    train_losses = []
    val_losses = []
    memory_usage = []
    peak_memory = 0
    epoch_times = []
    
    # Initial memory measurement
    initial_memory = get_memory_usage()
    memory_usage.append(initial_memory)
    
    # Initial evaluation
    model_copy.eval()
    val_loss = evaluate(model_copy, val_dataloader, device)
    val_losses.append(val_loss)
    print(f"Initial validation loss: {val_loss:.4f}")
    
    # Training loop
    model_copy.train()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        total_loss = 0
        
        # Progress bar for this epoch
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                           desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model_copy(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Scheduler step
            scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Measure memory every 50 batches
            if batch_idx % 50 == 0:
                mem_usage = get_memory_usage()
                memory_usage.append(mem_usage)
                
                if torch.cuda.is_available():
                    current_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    peak_memory = max(peak_memory, current_peak)
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Evaluate on validation set
        val_loss = evaluate(model_copy, val_dataloader, device)
        val_losses.append(val_loss)
        
        # Measure epoch time
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Time = {epoch_time:.2f}s")
    
    # Final memory measurement
    final_memory = get_memory_usage()
    memory_usage.append(final_memory)
    
    # Collect results
    results = {
        "method": method,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epoch_times": epoch_times,
        "peak_memory_gb": peak_memory,
        "memory_usage": memory_usage
    }
    
    # Clean up
    del model_copy, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return results

def plot_comparison(results, args):
    """Create plots comparing different methods"""
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    
    # Get method names and colors
    methods = [result["method"] for result in results]
    method_colors = {
        "blockllm": "blue",
        "galore": "orange",
        "full": "green"
    }
    
    # 1. Memory Usage Comparison
    plt.subplot(2, 2, 1)
    memory_data = [result["peak_memory_gb"] for result in results]
    method_names = [method.capitalize() if method != "full" else "Full Fine-tuning" for method in methods]
    
    bars = plt.bar(method_names, memory_data, color=[method_colors[m] for m in methods])
    plt.title("Peak GPU Memory Usage")
    plt.ylabel("Memory (GB)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f} GB', ha='center', va='bottom')
    
    # 2. Training Loss Convergence
    plt.subplot(2, 2, 2)
    for result in results:
        method = result["method"]
        epochs = range(len(result["train_losses"]))
        plt.plot(epochs, result["train_losses"], marker='o', linestyle='-', 
                 label=f"{method.capitalize() if method != 'full' else 'Full Fine-tuning'} (Train)",
                 color=method_colors[method])
    
    plt.title("Training Loss Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # 3. Validation Loss Convergence
    plt.subplot(2, 2, 3)
    for result in results:
        method = result["method"]
        epochs = range(len(result["val_losses"]))
        plt.plot(epochs, result["val_losses"], marker='s', linestyle='-', 
                 label=f"{method.capitalize() if method != 'full' else 'Full Fine-tuning'} (Val)",
                 color=method_colors[method])
    
    plt.title("Validation Loss Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # 4. Training Time Comparison
    plt.subplot(2, 2, 4)
    avg_epoch_times = [sum(result["epoch_times"])/len(result["epoch_times"]) for result in results]
    
    bars = plt.bar(method_names, avg_epoch_times, color=[method_colors[m] for m in methods])
    plt.title("Average Epoch Training Time")
    plt.ylabel("Time (seconds)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, "optimization_comparison.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(args.output_dir, "optimization_comparison.pdf"), bbox_inches="tight")
    
    print(f"Plots saved to {args.output_dir}")
    
    # Return the figure for display
    return plt.gcf()

def create_comparison_table(results):
    """Create a markdown table with comparison results"""
    table = "| Method | Final Train Loss | Final Val Loss | Avg Epoch Time | Peak Memory |\n"
    table += "|--------|-----------------|---------------|----------------|-------------|\n"
    
    for result in results:
        method = result["method"]
        method_name = method.capitalize() if method != "full" else "Full Fine-tuning"
        final_train_loss = result["train_losses"][-1]
        final_val_loss = result["val_losses"][-1]
        avg_epoch_time = sum(result["epoch_times"]) / len(result["epoch_times"])
        peak_memory = result["peak_memory_gb"]
        
        table += f"| {method_name} | {final_train_loss:.4f} | {final_val_loss:.4f} | {avg_epoch_time:.2f}s | {peak_memory:.2f} GB |\n"
    
    return table

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    
    # Parse methods to compare
    methods_to_compare = args.methods.split(",")
    print(f"Comparing methods: {methods_to_compare}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Ensure proper padding settings for auto-regressive models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load base model (will be copied for each method)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_data(tokenizer, args.dataset, max_length=args.max_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Train with each method and collect results
    all_results = []
    
    for method in methods_to_compare:
        if method not in ["blockllm", "galore", "full"]:
            print(f"Warning: Unknown method '{method}', skipping")
            continue
        
        result = train_model(model, method, train_dataloader, val_dataloader, args, device)
        all_results.append(result)
    
    # Create comparison plots
    plot_comparison(all_results, args)
    
    # Create comparison table
    comparison_table = create_comparison_table(all_results)
    print("\nComparison Results:\n")
    print(comparison_table)
    
    # Save table to file
    with open(os.path.join(args.output_dir, "comparison_results.md"), "w") as f:
        f.write("# BlockLLM vs GaLore Optimization Comparison\n\n")
        f.write("## Model: " + args.model_name + "\n")
        f.write("## Dataset: " + args.dataset + "\n\n")
        f.write(comparison_table)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()