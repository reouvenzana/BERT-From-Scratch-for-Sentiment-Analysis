import os
from typing import Optional
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, TrainerCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration
import numpy as np
import csv
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from models.bert2 import BERT  # Custom implementation of BERT from scratch using PyTorch

# Load datasets
train_df = pd.read_csv('./data/train_rotten_tomatoes_movie_reviews.csv')
val_df = pd.read_csv('./data/val_rotten_tomatoes_movie_reviews.csv')
test_df = pd.read_csv('./data/test_rotten_tomatoes_movie_reviews.csv')

# Use a smaller dataset for testing
train_df = train_df.head(1000)

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

#? Define Dataset class
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
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.argmax(-1)
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



class SavePthCallback(TrainerCallback):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        epoch = state.epoch
        if epoch is None:
            epoch = state.global_step // (state.max_steps // args.num_train_epochs)
        save_path = os.path.join(self.save_dir, f"model_epoch_{int(epoch)}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved model weights to {save_path}")


class EarlyStopping:
    def __init__(self, patience=2, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(size, train_encodings, train_labels, val_encodings, val_labels, test_encodings, test_labels, precision='bf16', output_results='./resultv2s', model_name='bert-base-uncased', finetuning=False):
    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)
    test_dataset = SentimentDataset(test_encodings, test_labels)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sizes_to_process = ['small', 'base', 'large'] if size is None else [size]

    for size in sizes_to_process:
        if not finetuning:
            # Configuration du modèle BERT from scratch
            if size == 'tiny':
                embed_size, num_layers, heads = 128, 2, 2
            elif size == 'small':
                embed_size, num_layers, heads = 256, 4, 8
            elif size == 'base':
                embed_size, num_layers, heads = 768, 12, 8
            elif size == 'large':
                embed_size, num_layers, heads = 1024, 24, 16
            else:
                raise ValueError("Size must be one of 'tiny', 'small', 'base', or 'large' for training from scratch")

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

            optimizer = AdamW(model.parameters(), lr=4e-5, betas=(0.9, 0.95), weight_decay=0.1)
            accelerator = Accelerator(mixed_precision='no')

            train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=32)
            test_dataloader = DataLoader(test_dataset, batch_size=32)

            model, optimizer, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, val_dataloader, test_dataloader
            )

            if precision == 'bf16':
                model.to(torch.bfloat16)
                expected_dtype = torch.bfloat16
            elif precision == 'fp32':
                model.to(torch.float32)
                expected_dtype = torch.float32
            else:
                raise ValueError("Precision must be either 'bf16' or 'fp32'")

            check_model_dtype(model, expected_dtype)

            num_epochs = 20
            num_training_steps = num_epochs * len(train_dataloader)
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=5e-6)

            check_optimizer_dtype(optimizer, expected_dtype)

            result_dir = f'{output_results}/from_scratch/bert-{size}-{precision}'
            os.makedirs(result_dir, exist_ok=True)

            metrics_file = f'{result_dir}/training_metrics.csv'
            with open(metrics_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch', 'val_loss', 'val_accuracy', 'val_f1', 'val_precision', 'val_recall'])

            loss_file = f'{result_dir}/loss.csv'
            with open(loss_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch', 'batch', 'train_loss', 'val_loss'])

            early_stopping = EarlyStopping(patience=1, min_delta=0.01)
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

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    progress_bar.set_postfix(loss=loss.item())

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

                        with open(loss_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([epoch+1, batch_num, '', loss.item()])
                        batch_num += 1

                val_preds = torch.cat(val_preds).to('cpu')
                val_labels_list = torch.cat(val_labels_list).to('cpu')
                eval_pred = EvalPrediction(predictions=val_preds, label_ids=val_labels_list)
                val_metrics = compute_metrics(eval_pred)
                val_loss /= len(val_dataloader)
                train_loss /= len(train_dataloader)

                if early_stopping(val_loss):
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

                with open(metrics_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch+1, val_loss, val_metrics['accuracy'], val_metrics['f1'], val_metrics['precision'], val_metrics['recall']])

                with open(loss_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch+1, '', train_loss, val_loss])

                model.train()

            model.eval()
            test_loss = 0
            test_preds = []
            test_labels_list = []

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
            eval_pred = EvalPrediction(predictions=test_preds, label_ids=test_labels_list)
            test_metrics = compute_metrics(eval_pred)
            test_loss /= len(test_dataloader)

            test_metrics_file = f'{result_dir}/test_metrics.csv'
            with open(test_metrics_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['test_loss', 'test_accuracy', 'test_f1', 'test_precision', 'test_recall'])
                writer.writerow([test_loss, test_metrics['accuracy'], test_metrics['f1'], test_metrics['precision'], test_metrics['recall']])

            with open(loss_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['final_test', '', '', '', test_loss])

        else:
            print("Finetuning started!")
            if size == 'small':
                model_name = 'prajjwal1/bert-small'
            elif size == 'base':
                model_name = 'bert-base-uncased'
            elif size == 'large':
                model_name = 'bert-large-uncased'
            else:
                raise ValueError("Size must be one of 'small', 'base', or 'large' for fine-tuning")

            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
            
            finetuning_base_dir = f'{output_results}/finetuning'
            finetuning_dir = f'{finetuning_base_dir}/finetuning-{size}'
            os.makedirs(finetuning_dir, exist_ok=True)

            # Créer les fichiers CSV pour enregistrer les métriques et la loss
            metrics_file = f'{finetuning_dir}/training_metrics.csv'
            with open(metrics_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_f1', 'val_precision', 'val_recall'])

            loss_file = f'{finetuning_dir}/loss.csv'
            with open(loss_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch', 'batch', 'train_loss', 'val_loss'])

            training_args = TrainingArguments(
                output_dir=finetuning_dir,
                num_train_epochs=5,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
                learning_rate=2e-5,
                weight_decay=0.01,
                evaluation_strategy="epoch",
                save_strategy="no",  # Désactiver la sauvegarde automatique
                load_best_model_at_end=False, 
                logging_dir=f'{finetuning_dir}/logs',
                logging_steps=10,
                report_to="none",
                save_safetensors=False
            )
            
            # Instancier le callback personnalisé
            save_pth_callback = SavePthCallback(save_dir=finetuning_dir)
            
            class CustomTrainer(Trainer):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.train_loss = 0
                    self.train_batch_count = 0
                    self.metrics_file = f'{self.args.output_dir}/training_metrics.csv'
                    self.loss_file = f'{self.args.output_dir}/loss.csv'
                    
                    # Initialiser les fichiers CSV
                    with open(self.metrics_file, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_f1', 'val_precision', 'val_recall'])
                    
                    with open(self.loss_file, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['epoch', 'batch', 'train_loss', 'val_loss'])

                def training_step(self, model, inputs):
                    loss = super().training_step(model, inputs)
                    self.train_loss += loss.item()
                    self.train_batch_count += 1
                    
                    # Enregistrer la loss pour chaque batch
                    with open(self.loss_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([self.state.epoch, self.state.global_step, loss.item(), ''])
                    
                    return loss

                def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
                    # Calcul de la perte moyenne d'entraînement
                    avg_train_loss = self.train_loss / self.train_batch_count if self.train_batch_count > 0 else 0
                    
                    # Réinitialisation des compteurs pour la prochaine époque
                    self.train_loss = 0
                    self.train_batch_count = 0

                    # Appel de la méthode d'évaluation parent
                    metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

                    # Ajout de la perte moyenne d'entraînement aux métriques
                    metrics['train_loss'] = avg_train_loss

                    # Enregistrer les métriques dans le fichier CSV
                    with open(self.metrics_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            self.state.epoch,
                            avg_train_loss,
                            metrics['eval_loss'],
                            metrics['eval_accuracy'],
                            metrics['eval_f1'],
                            metrics['eval_precision'],
                            metrics['eval_recall']
                        ])

                    # Enregistrer la loss de validation
                    with open(self.loss_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([self.state.epoch, '', '', metrics['eval_loss']])

                    return metrics

                def save_model(self, output_dir=None, _internal_call=False):
                    if output_dir is None:
                        output_dir = self.args.output_dir
                    
                    # Sauvegarde du modèle au format .pth
                    torch.save(self.model.state_dict(), f"{output_dir}/model.pth")
                    
                    # Sauvegarde de la configuration
                    self.model.config.save_pretrained(output_dir)
                    
                    # Sauvegarde des arguments d'entraînement
                    torch.save(self.args, f"{output_dir}/training_args.bin")

                def _save(self, output_dir: Optional[str] = None, state_dict=None):
                    # Override cette méthode pour utiliser votre logique de sauvegarde personnalisée
                    self.save_model(output_dir)

            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[save_pth_callback]  # Ajouter le callback personnalisé ici
            )

            trainer.train()

            # Évaluer sur l'ensemble de test
            test_metrics = trainer.evaluate(test_dataset)

            test_metrics_file = f'{finetuning_dir}/test_metrics.csv'
            with open(test_metrics_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['test_loss', 'test_accuracy', 'test_f1', 'test_precision', 'test_recall'])
                writer.writerow([
                    test_metrics['eval_loss'],
                    test_metrics['eval_accuracy'],
                    test_metrics['eval_f1'],
                    test_metrics['eval_precision'],
                    test_metrics['eval_recall']
                ])

            print(f"Finetuning results for {size} BERT:", test_metrics)


# Préparer les datasets
train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)


#& Fine-tuning des modèles pré-entraînés
for size in ['small', 'base', 'large']:
    print(f"Fine-tuning BERT model of size {size}")
    train(
        size=size,
        train_encodings=train_encodings,
        train_labels=train_labels,
        val_encodings=val_encodings,
        val_labels=val_labels,
        test_encodings=test_encodings,
        test_labels=test_labels,
        precision='fp32', 
        output_results='./Test7',
        finetuning=True
    )

print("finetuning fini")

#& Entraînement from scratch
for size in ['tiny', 'small', 'base', 'large']:
    for precision in ['fp32', 'bf16']:
        print(f"Training from scratch: size={size}, precision={precision}")
        train(
            size=size,
            train_encodings=train_encodings,
            train_labels=train_labels,
            val_encodings=val_encodings,
            val_labels=val_labels,
            test_encodings=test_encodings,
            test_labels=test_labels,
            precision=precision,
            output_results="Test7",
            finetuning=False
        )