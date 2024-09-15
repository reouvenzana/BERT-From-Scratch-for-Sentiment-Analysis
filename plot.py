import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de Seaborn
sns.set(style="whitegrid")

def moving_average(data, window_size=100):
    """
    Applique une moyenne mobile sur les données fournies.
    """
    return data.rolling(window=window_size, min_periods=1).mean()

def plot_loss(run_path, plots_dir, model_name, mode):
    """
    Génère un graphique des pertes d'entraînement et de validation.
    La train loss est lissée avec une moyenne mobile.
    """
    loss_train_file = os.path.join(run_path, 'loss_train.csv')
    loss_valset_file = os.path.join(run_path, 'loss_valset.csv')
    
    if not os.path.exists(loss_train_file) or not os.path.exists(loss_valset_file):
        print(f"Fichiers de perte manquants pour {model_name} dans {run_path}. Skipping Loss Plot...")
        return
    
    try:
        # Lire les fichiers CSV
        loss_train_df = pd.read_csv(loss_train_file)
        loss_valset_df = pd.read_csv(loss_valset_file)
        
        # Vérifier la présence des colonnes nécessaires
        if not {'epoch', 'value'}.issubset(loss_train_df.columns) or not {'epoch', 'value'}.issubset(loss_valset_df.columns):
            print(f"Format incorrect des fichiers de perte pour {model_name}. Skipping Loss Plot...")
            return
        
        # Appliquer une moyenne mobile à la train loss
        loss_train_df['loss_smooth'] = moving_average(loss_train_df['value'], window_size=100)
        
        plt.figure(figsize=(12, 7))
        
        # Tracer la train loss lissée
        plt.plot(loss_train_df['epoch'], loss_train_df['loss_smooth'], label='Train Loss (Smoothed)', color='blue', alpha=0.7)
        
        # Tracer la val loss
        plt.plot(loss_valset_df['epoch'], loss_valset_df['value'], label='Validation Loss', color='red', marker='o')
        
        plt.xlabel('Époque')
        plt.ylabel('Loss')
        plt.title(f'{model_name} - {mode.capitalize()} - Loss par Époque')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{model_name}_{mode}_loss.png'))
        plt.close()
        
        print(f"Graphique des Loss généré pour {model_name} dans le mode {mode}.")
        
    except Exception as e:
        print(f"Erreur lors de la génération du graphique des Loss pour {model_name} : {e}")

def plot_metrics(run_path, plots_dir, model_name, mode):
    """
    Génère un graphique des métriques (accuracy, F1, precision, recall).
    """
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    datasets = ['valset']  # Vous pouvez ajouter 'train' si les métriques d'entraînement sont disponibles
    
    plt.figure(figsize=(12, 7))
    
    for metric in metrics:
        for dataset in datasets:
            metric_file = os.path.join(run_path, f"{metric}_{dataset}.csv")
            if os.path.exists(metric_file):
                try:
                    df = pd.read_csv(metric_file)
                    if not {'epoch', 'value'}.issubset(df.columns):
                        print(f"Format incorrect du fichier de métrique {metric_file}. Skipping...")
                        continue
                    plt.plot(df['epoch'], df['value'], label=f'{metric.capitalize()} ({dataset})', marker='o')
                except Exception as e:
                    print(f"Erreur lors de la lecture de {metric_file} : {e}")
            else:
                print(f"Fichier de métrique manquant : {metric_file}")
    
    plt.xlabel('Époque')
    plt.ylabel('Valeur')
    plt.title(f'{model_name} - {mode.capitalize()} - Métriques par Époque')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_{mode}_metrics.png'))
    plt.close()
    
    print(f"Graphique des Métriques généré pour {model_name} dans le mode {mode}.")

def process_run(run_path, mode, plots_dir):
    """
    Traite un run spécifique pour générer les graphiques de Loss et de Métriques.
    """
    model_name = os.path.basename(run_path)
    print(f"Traitement de {model_name} dans {run_path}")
    
    # Générer le plot des Loss
    plot_loss(run_path, plots_dir, model_name, mode)
    
    # Générer le plot des Métriques
    plot_metrics(run_path, plots_dir, model_name, mode)

def main():
    base_output_dir = './reel_1'
    plots_dir = os.path.join(base_output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Traitement du dossier from_scratch
    from_scratch_dir = os.path.join(base_output_dir, 'from_scratch')
    if os.path.exists(from_scratch_dir):
        for model_name in os.listdir(from_scratch_dir):
            run_path = os.path.join(from_scratch_dir, model_name)
            if os.path.isdir(run_path):
                process_run(run_path, 'from_scratch', plots_dir)
    else:
        print(f"Répertoire {from_scratch_dir} n'existe pas. Skipping from_scratch...")

    # Traitement du dossier finetuning
    finetuning_dir = os.path.join(base_output_dir, 'finetuning')
    if os.path.exists(finetuning_dir):
        for model_type in os.listdir(finetuning_dir):
            model_type_path = os.path.join(finetuning_dir, model_type)
            if os.path.isdir(model_type_path):
                process_run(model_type_path, 'finetuning', plots_dir)
    else:
        print(f"Répertoire {finetuning_dir} n'existe pas. Skipping finetuning...")

    print("Tous les graphiques ont été générés dans le dossier 'plots'.")

if __name__ == "__main__":
    main()