# src/visualize_results.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

results_dir = 'results'
result_img_dir = 'result_img'

def get_save_directory():
    """Get appropriate directory for saving plots"""
    if os.path.exists(results_dir) and os.access(results_dir, os.W_OK):
        return results_dir
    else:
        try:
            os.makedirs(result_img_dir, exist_ok=True)
            return result_img_dir
        except Exception as e:
            print(f"Error creating directories: {e}")
            return '.'

def save_plot(plt, filename):
    """Safely save plot to file"""
    save_dir = get_save_directory()
    try:
        full_path = os.path.join(save_dir, filename)
        plt.savefig(full_path)
        print(f"Successfully saved: {full_path}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")
        try:
            plt.savefig(filename)
            print(f"Saved in current directory: {filename}")
        except Exception as e:
            print(f"Failed to save plot anywhere: {e}")

def visualize_dataset(dataset_name, max_samples=10000):
    """Visualize clustering results for any dataset"""
    try:
        # Load data
        if dataset_name == 'covertype':
            features = pd.read_csv('data/covertype_features.csv')
            targets = pd.read_csv('data/covertype_target.csv')
            targets = targets['Cover_Type'].values
        elif dataset_name == 'mnist':  # Change back to mnist since files are mnist_*
            features = pd.read_csv('data/mnist_features.csv')
            targets = pd.read_csv('data/mnist_target.csv')
            targets = targets.iloc[:, 0].values
        else:  # creditcard
            df = pd.read_csv('data/creditcard.csv')
            features = df.drop(['Class', 'Time'], axis=1)
            targets = df['Class'].values

        # Convert features to numpy array if it's a DataFrame
        features = features.values if isinstance(features, pd.DataFrame) else features
        
        # Load clustering results first to determine if we need to sample
        centroids = np.load(os.path.join(results_dir, f'{dataset_name}_centroids.npy'))
        pred_labels = np.load(os.path.join(results_dir, f'{dataset_name}_labels.npy'))
        
        # Sample if dataset is too large - use same indices for all data
        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]
            targets = targets[indices]
            pred_labels = pred_labels[indices]  # Sample the predictions too
        
        # Dimensionality reduction
        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2, random_state=42)
        
        X_pca = pca.fit_transform(features)
        X_tsne = tsne.fit_transform(features)
        centroids_pca = pca.transform(centroids)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
        
        # PCA plots
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=targets.astype(int), cmap='viridis', alpha=0.6)
        ax1.set_title(f'{dataset_name} - PCA (True Labels)')
        plt.colorbar(scatter1, ax=ax1)
        
        scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=pred_labels, cmap='viridis', alpha=0.6)
        ax2.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='x', s=200, linewidths=3)
        ax2.set_title(f'{dataset_name} - PCA (Predicted Clusters)')
        plt.colorbar(scatter2, ax=ax2)
        
        # t-SNE plots
        scatter3 = ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=targets.astype(int), cmap='viridis', alpha=0.6)
        ax3.set_title(f'{dataset_name} - t-SNE (True Labels)')
        plt.colorbar(scatter3, ax=ax3)
        
        scatter4 = ax4.scatter(X_tsne[:, 0], X_tsne[:, 1], c=pred_labels, cmap='viridis', alpha=0.6)
        ax4.set_title(f'{dataset_name} - t-SNE (Predicted Clusters)')
        plt.colorbar(scatter4, ax=ax4)
        
        plt.tight_layout()
        save_plot(plt, f'{dataset_name}_visualization.png')
        plt.close()
        
        # Create confusion matrix heatmap
        plt.figure(figsize=(10, 8))
        confusion_matrix = pd.crosstab(targets, pred_labels)
        sns.heatmap(confusion_matrix, annot=True, cmap='YlOrRd')
        plt.title(f'{dataset_name} - Clustering vs True Labels')
        save_plot(plt, f'{dataset_name}_confusion_matrix.png')
        plt.close()
        
    except Exception as e:
        print(f"Error in {dataset_name} visualization: {str(e)}")

if __name__ == "__main__":
    save_dir = get_save_directory()
    print(f"Using directory for saving plots: {save_dir}")
    
    for dataset in ['covertype', 'mnist', 'creditcard']:  # Changed back to mnist
        print(f"\nGenerating {dataset} visualization...")
        visualize_dataset(dataset)