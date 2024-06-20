import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, get_scheduler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration
import csv
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from models.bert import BERT  # Custom implementation of BERT from scratch using PyTorch

# Load datasets
train_df = pd.read_csv('./data/train_rotten_tomatoes_movie_reviews.csv')
val_df = pd.read_csv('./data/val_rotten_tomatoes_movie_reviews.csv')
test_df = pd.read_csv('./data/test_rotten_tomatoes_movie_reviews.csv')

# Use a smaller dataset for testing
train_df = train_df.head(2000)

# Preprocess data
train_texts = train_df['reviewText'].tolist()
train_labels = train_df['label'].tolist()
val_texts = val_df['reviewText'].tolist()
val_labels = val_df['label'].tolist()
test_texts = test_df['reviewText'].tolist()
test_labels = test_df['label'].tolist()

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Define Dataset class
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Define metrics computation function
def compute_metrics(preds, labels):
    preds = preds.argmax(-1)
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def check_model_dtype(model, expected_dtype):
    for name, param in model.named_parameters():
        if param.dtype != expected_dtype:
            print(f"Parameter {name} is not in {expected_dtype}, it is in {param.dtype}")
        assert param.dtype == expected_dtype, f"Parameter {name} is not in {expected_dtype}, it is in {param.dtype}"
    for name, buffer in model.named_buffers():
        if buffer.dtype != expected_dtype:
            print(f"Buffer {name} is not in {expected_dtype}, it is in {buffer.dtype}")
        assert buffer.dtype == expected_dtype, f"Buffer {name} is not in {expected_dtype}, it is in {buffer.dtype}"
    print("All model parameters and buffers are in the expected dtype.")

def check_optimizer_dtype(optimizer, expected_dtype):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.dtype != expected_dtype:
                print(f"Optimizer parameter is not in {expected_dtype}, it is in {param.dtype}")
            assert param.dtype == expected_dtype, f"Optimizer parameter is not in {expected_dtype}, it is in {param.dtype}"
    print("All optimizer parameters are in the expected dtype.")

def train(size, train_encodings, train_labels, val_encodings, val_labels, test_encodings, test_labels):
    # Prepare datasets and dataloaders
    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)
    test_dataset = SentimentDataset(test_encodings, test_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model configuration based on size
    if size == 'tiny':
        embed_size = 128
        num_layers = 2
        heads = 4
    elif size == 'small':
        embed_size = 256
        num_layers = 4
        heads = 8
    elif size == 'base':
        embed_size = 768
        num_layers = 12
        heads = 8
    elif size == 'large':
        embed_size = 1024
        num_layers = 24
        heads = 16

    model = BERT(
        vocab_size=30522,
        embed_size=embed_size,
        num_layers=num_layers,
        heads=heads,
        device=device,
        forward_expansion=4,
        dropout=0.1,
        max_length=512
    ).to(device)

    # Apply TorchScript for model optimization
    model = torch.jit.script(model)
    
    # Initialize Accelerator with mixed precision
    dataloader_config = DataLoaderConfiguration(dispatch_batches=None, split_batches=False)
    accelerator = Accelerator(dataloader_config=dataloader_config, mixed_precision="bf16")

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128)
    test_dataloader = DataLoader(test_dataset, batch_size=128)

    model, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(model, train_dataloader, val_dataloader, test_dataloader)

    # Convert model to bfloat16
    model.to(torch.bfloat16)

    # Check if model parameters are in bf16
    check_model_dtype(model, torch.bfloat16)

    optimizer = AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0.1)
    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    
    # Use CosineAnnealingLR scheduler with final learning rate 10% of max
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=5e-6)

    # Check if optimizer parameters are in bf16
    check_optimizer_dtype(optimizer, torch.bfloat16)

    # Create folder for saving results
    result_dir = f'./results/bert-{size}'
    os.makedirs(result_dir, exist_ok=True)

    # Save training metrics
    metrics_file = f'{result_dir}/training_metrics.csv'
    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'val_loss', 'val_accuracy', 'val_f1', 'val_precision', 'val_recall'])

    # Save detailed loss metrics
    loss_file = f'{result_dir}/loss.csv'
    with open(loss_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'batch', 'train_loss', 'val_loss'])

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        train_loss = 0
        batch_num = 0

        for batch in progress_bar:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = F.cross_entropy(outputs, batch['labels'])
            train_loss += loss.item()
            accelerator.backward(loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())

            # Save batch train loss
            with open(loss_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch+1, batch_num, loss.item(), ''])
            batch_num += 1

        torch.save(model.state_dict(), f'{result_dir}/model_epoch_{epoch+1}.pth')

        model.eval()
        val_loss = 0
        val_preds = []
        val_labels_list = []
        batch_num = 0

        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                loss = F.cross_entropy(outputs, batch['labels'])
                val_loss += loss.item()
                logits = outputs
                labels = batch['labels']
                val_preds.append(logits)
                val_labels_list.append(labels)

                # Save batch validation loss
                with open(loss_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch+1, batch_num, '', loss.item()])
                batch_num += 1

        val_preds = torch.cat(val_preds).to('cpu')
        val_labels_list = torch.cat(val_labels_list).to('cpu')
        val_metrics = compute_metrics(val_preds, val_labels_list)
        val_loss /= len(val_dataloader)
        train_loss /= len(train_dataloader)

        # Save epoch validation metrics
        with open(metrics_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, val_loss, val_metrics['accuracy'], val_metrics['f1'], val_metrics['precision'], val_metrics['recall']])

        # Save epoch train and validation loss
        with open(loss_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, '', train_loss, val_loss])

        model.train()

    # Final evaluation on the test set after training
    model.eval()
    test_loss = 0
    test_preds = []
    test_labels_list = []
    batch_num = 0

    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = F.cross_entropy(outputs, batch['labels'])
            test_loss += loss.item()
            logits = outputs
            labels = batch['labels']
            test_preds.append(logits)
            test_labels_list.append(labels)

    test_preds = torch.cat(test_preds).to('cpu')
    test_labels_list = torch.cat(test_labels_list).to('cpu')
    test_metrics = compute_metrics(test_preds, test_labels_list)
    test_loss /= len(test_dataloader)

    # Save test metrics
    test_metrics_file = f'{result_dir}/test_metrics.csv'
    with open(test_metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['test_loss', 'test_accuracy', 'test_f1', 'test_precision', 'test_recall'])
        writer.writerow([test_loss, test_metrics['accuracy'], test_metrics['f1'], test_metrics['precision'], test_metrics['recall']])

    # Save final test loss
    with open(loss_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['final_test', '', '', '', test_loss])

# Train models of different sizes
for size in ['tiny', 'small', 'base', 'large']:  # 'tiny', 'small', 'base', 'large'
    train(size, train_encodings, train_labels, val_encodings, val_labels, test_encodings, test_labels)