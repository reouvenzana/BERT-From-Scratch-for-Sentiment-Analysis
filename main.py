import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration

# Charger les données
df = pd.read_csv('./../data/rotten_tomatoes_movie_reviews.csv') #? https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews
# df = df.head(2000) #? for tgesting purposes

# Prétraiter les données
df = df[['reviewText', 'scoreSentiment']]
df = df.dropna()
df['label'] = df['scoreSentiment'].apply(lambda x: 1 if x == 'POSITIVE' else 0)

train_texts, val_texts, train_labels, val_labels = train_test_split(df['reviewText'].tolist(), df['label'].tolist(), test_size=0.025)

# Tokenisation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

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

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

dataloader_config = DataLoaderConfiguration(dispatch_batches=None, split_batches=False)
accelerator = Accelerator(dataloader_config=dataloader_config)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)

model, train_dataloader, val_dataloader = accelerator.prepare(model, train_dataloader, val_dataloader)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps
)

model.train()

for epoch in range(num_epochs):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch in progress_bar:
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix(loss=loss.item())
    
    torch.save(model.state_dict(), f'./results/model_epoch_{epoch+1}.pth')

model.eval()

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

eval_progress_bar = tqdm(val_dataloader, desc="Evaluating")
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in eval_progress_bar:
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        labels = batch['labels']
        
        all_preds.append(logits)
        all_labels.append(labels)

all_preds = torch.cat(all_preds).to('cpu')
all_labels = torch.cat(all_labels).to('cpu')

metrics = compute_metrics(all_preds, all_labels)
print(metrics)