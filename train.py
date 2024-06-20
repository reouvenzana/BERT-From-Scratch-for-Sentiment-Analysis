import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration
from models.bert import BERT
import csv

# Load datasets
train_df = pd.read_csv('./data/train_rotten_tomatoes_movie_reviews.csv')
val_df = pd.read_csv('./data/val_rotten_tomatoes_movie_reviews.csv')
test_df = pd.read_csv('./data/test_rotten_tomatoes_movie_reviews.csv')

# Use a smaller dataset for testing
train_df = train_df.head(3000)

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
class SentimentDataset(torch.utils.data.Dataset):
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

def train(size, train_encodings, train_labels, val_encodings, val_labels, test_encodings, test_labels):
    # Prepare datasets and dataloaders
    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)
    test_dataset = SentimentDataset(test_encodings, test_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model configuration based on size
    if size == 'small':
        embed_size = 256
        num_layers = 4
        heads = 4
    elif size == 'medium':
        embed_size = 512
        num_layers = 8
        heads = 8
    elif size == 'big':
        embed_size = 768
        num_layers = 12
        heads = 12

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

    dataloader_config = DataLoaderConfiguration(dispatch_batches=None, split_batches=False)
    accelerator = Accelerator(dataloader_config=dataloader_config)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    model, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(model, train_dataloader, val_dataloader, test_dataloader)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps
    )

    # Create folder for saving results
    result_dir = f'./results/bert-{size}'
    os.makedirs(result_dir, exist_ok=True)

    # Save training metrics
    metrics_file = f'{result_dir}/training_metrics.csv'
    with open(metrics_file, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_f1', 'val_precision', 'val_recall'])

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        train_loss = 0

        for batch in progress_bar:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = F.cross_entropy(outputs, batch['labels'])
            train_loss += loss.item()
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())

        torch.save(model.state_dict(), f'{result_dir}/model_epoch_{epoch+1}.pth')

        model.eval()
        val_loss = 0
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                val_loss += F.cross_entropy(outputs, batch['labels']).item()
                logits = outputs
                labels = batch['labels']
                val_preds.append(logits)
                val_labels_list.append(labels)

        val_preds = torch.cat(val_preds).to('cpu')
        val_labels_list = torch.cat(val_labels_list).to('cpu')
        val_metrics = compute_metrics(val_preds, val_labels_list)
        val_loss /= len(val_dataloader)
        train_loss /= len(train_dataloader)

        with open(metrics_file, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, train_loss, val_loss, val_metrics['accuracy'], val_metrics['f1'], val_metrics['precision'], val_metrics['recall']])

        model.train()

# Train models of different sizes
for size in ['small', 'medium', 'big']:
    train(size, train_encodings, train_labels, val_encodings, val_labels, test_encodings, test_labels)

#* first training : {'accuracy': 0.9074257137872885, 'f1': 0.9314531754574812, 'precision': 0.9266255461320997, 'recall': 0.9363313711911357}