import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration

df = pd.read_csv('./data/rotten_tomatoes_movie_reviews.csv') #? https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews
#! testing with a smaller dataset
# df = df.head(2000) #? for tgesting purposes

df = df[['reviewText', 'scoreSentiment']]
df = df.dropna()
df['label'] = df['scoreSentiment'].apply(lambda x: 1 if x == 'POSITIVE' else 0)

train_texts, val_texts, train_labels, val_labels = train_test_split(df['reviewText'].tolist(), df['label'].tolist(), test_size=0.025)

#! try another tokenizer
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








# tiny_bert = {
#     "vocab_size": 30522,           # Standard BERT vocabulary size
#     "embed_size": 128,             # Smaller embedding size
#     "num_layers": 2,               # Fewer transformer layers
#     "heads": 2,                    # Fewer attention heads
#     "forward_expansion": 4,        # Expansion rate in feed-forward network
#     "dropout": 0.1,                # Dropout rate
#     "max_length": 512              # Maximum sequence length
# }

# # Small BERT
# small_bert = {
#     "vocab_size": 30522,           # Standard BERT vocabulary size
#     "embed_size": 256,             # Smaller embedding size
#     "num_layers": 4,               # Fewer transformer layers
#     "heads": 4,                    # Fewer attention heads
#     "forward_expansion": 4,        # Expansion rate in feed-forward network
#     "dropout": 0.1,                # Dropout rate
#     "max_length": 512              # Maximum sequence length
# }

# # Medium BERT
# medium_bert = {
#     "vocab_size": 30522,           # Standard BERT vocabulary size
#     "embed_size": 512,             # Standard embedding size
#     "num_layers": 8,               # Standard transformer layers count
#     "heads": 8,                    # Standard attention heads count
#     "forward_expansion": 4,        # Expansion rate in feed-forward network
#     "dropout": 0.1,                # Dropout rate
#     "max_length": 512              # Maximum sequence length
# }

# # Large BERT
# large_bert = {
#     "vocab_size": 30522,           # Standard BERT vocabulary size
#     "embed_size": 1024,            # Larger embedding size
#     "num_layers": 16,              # More transformer layers
#     "heads": 16,                   # More attention heads
#     "forward_expansion": 4,        # Expansion rate in feed-forward network
#     "dropout": 0.1,                # Dropout rate
#     "max_length": 512              # Maximum sequence length
# }

# # Huge BERT
# huge_bert = {
#     "vocab_size": 30522,           # Standard BERT vocabulary size
#     "embed_size": 2048,            # Even larger embedding size
#     "num_layers": 32,              # Even more transformer layers
#     "heads": 32,                   # Even more attention heads
#     "forward_expansion": 4,        # Expansion rate in feed-forward network
#     "dropout": 0.1,                # Dropout rate
#     "max_length": 512              # Maximum sequence length
# }

# # Example instantiation of a Tiny BERT model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = BERT(
#     vocab_size=tiny_bert["vocab_size"],
#     embed_size=tiny_bert["embed_size"],
#     num_layers=tiny_bert["num_layers"],
#     heads=tiny_bert["heads"],
#     device=device,
#     forward_expansion=tiny_bert["forward_expansion"],
#     dropout=tiny_bert["dropout"],
#     max_length=tiny_bert["max_length"]
# ).to(device)




















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



def predict_review(review_text):
    model.eval()
    inputs = tokenizer(review_text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(accelerator.device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).cpu().item()
    sentiment = 'POSITIVE' if prediction == 1 else 'NEGATIVE'
    return sentiment

#? example review
example_review = "This movie was absolutely fantastic! The performances were stellar, and the storyline was gripping."
predicted_sentiment = predict_review(example_review)
print(f"Review: {example_review}")
print(f"Predicted Sentiment: {predicted_sentiment}")
