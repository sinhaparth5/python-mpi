# src/visualize_results.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples

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

def load_dataset(dataset_name):
    """Load and preprocess dataset"""
    if dataset_name == 'covertype':
        features = pd.read_csv('data/covertype_features.csv')
        targets = pd.read_csv('data/covertype_target.csv')
        targets = targets['Cover_Type'].values
    elif dataset_name == 'mnist':
        features = pd.read_csv('data/mnist_features.csv')
        targets = pd.read_csv('data/mnist_target.csv')
        targets = targets.iloc[:, 0].values
    else:  # creditcard
        df = pd.read_csv('data/creditcard.csv')
        features = df.drop(['Class', 'Time'], axis=1)
        targets = df['Class'].values
    
    return features.values if isinstance(features, pd.DataFrame) else features, targets

def create_dimension_reduction_plots(features, targets, pred_labels, centroids, dataset_name):
    """Create PCA and t-SNE visualizations"""
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, random_state=42)
    
    X_pca = pca.fit_transform(features)
    X_tsne = tsne.fit_transform(features)
    centroids_pca = pca.transform(centroids)
    
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

def create_cluster_analysis_plots(features, targets, pred_labels, centroids, dataset_name):
    """Create cluster analysis visualizations"""
    # Cluster Size Distribution
    plt.figure(figsize=(12, 6))
    cluster_sizes = pd.Series(pred_labels).value_counts().sort_index()
    sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values)
    plt.title(f'{dataset_name} - Cluster Size Distribution')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    save_plot(plt, f'{dataset_name}_cluster_sizes.png')
    plt.close()
    
    # Cluster Characteristics Heatmap
    cluster_means = pd.DataFrame([
        features[pred_labels == i].mean(axis=0) 
        for i in range(len(np.unique(pred_labels)))
    ])
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(cluster_means, cmap='coolwarm', center=0)
    plt.title(f'{dataset_name} - Cluster Characteristics')
    plt.xlabel('Features')
    plt.ylabel('Clusters')
    save_plot(plt, f'{dataset_name}_cluster_characteristics.png')
    plt.close()
    
    # Inter-cluster Distance Matrix
    distances = np.zeros((len(centroids), len(centroids)))
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            distances[i,j] = np.linalg.norm(centroids[i] - centroids[j])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(distances, annot=True, cmap='YlGnBu')
    plt.title(f'{dataset_name} - Inter-cluster Distances')
    save_plot(plt, f'{dataset_name}_cluster_distances.png')
    plt.close()

def create_quality_metrics_plots(features, targets, pred_labels, dataset_name):
    """Create clustering quality visualizations"""
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    confusion_matrix = pd.crosstab(targets, pred_labels)
    sns.heatmap(confusion_matrix, annot=True, cmap='YlOrRd')
    plt.title(f'{dataset_name} - Clustering vs True Labels')
    save_plot(plt, f'{dataset_name}_confusion_matrix.png')
    plt.close()
    
    # Silhouette Plot
    silhouette_avg = silhouette_score(features, pred_labels)
    sample_silhouette_values = silhouette_samples(features, pred_labels)
    
    plt.figure(figsize=(10, 6))
    y_lower = 10
    for i in range(len(np.unique(pred_labels))):
        ith_cluster_values = sample_silhouette_values[pred_labels == i]
        ith_cluster_values.sort()
        
        size_cluster_i = len(ith_cluster_values)
        y_upper = y_lower + size_cluster_i
        
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_values,
                         alpha=0.7)
        y_lower = y_upper + 10
    
    plt.title(f'{dataset_name} - Silhouette Plot (avg: {silhouette_avg:.3f})')
    plt.xlabel('Silhouette Coefficient')
    plt.ylabel('Cluster')
    save_plot(plt, f'{dataset_name}_silhouette.png')
    plt.close()

def create_dataset_specific_plots(features, targets, pred_labels, dataset_name):
    """Create dataset-specific visualizations"""
    if dataset_name == 'creditcard':
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(x=targets, y=pred_labels)
        plt.title('Predicted Clusters by True Class')
        plt.xlabel('True Class (0=Normal, 1=Fraud)')
        plt.ylabel('Predicted Cluster')
        
        plt.subplot(1, 2, 2)
        detection_rates = pd.crosstab(targets, pred_labels, normalize='index')
        sns.heatmap(detection_rates, annot=True, cmap='YlOrRd')
        plt.title('Detection Rate by Class')
        
        save_plot(plt, f'{dataset_name}_specific_analysis.png')
        plt.close()
    
    elif dataset_name == 'mnist':
        plt.figure(figsize=(20, 8))
        for i in range(10):
            plt.subplot(2, 5, i+1)
            cluster_avg = features[pred_labels == i].mean(axis=0)
            plt.imshow(cluster_avg.reshape(28, 28), cmap='gray')
            plt.title(f'Cluster {i}')
        plt.suptitle('Average Digit by Cluster')
        save_plot(plt, f'{dataset_name}_cluster_averages.png')
        plt.close()

def visualize_dataset(dataset_name, max_samples=10000):
    """Main visualization function"""
    try:
        # Load data
        features, targets = load_dataset(dataset_name)
        
        # Load clustering results
        centroids = np.load(os.path.join(results_dir, f'{dataset_name}_centroids.npy'))
        pred_labels = np.load(os.path.join(results_dir, f'{dataset_name}_labels.npy'))
        
        # Sample if dataset is too large
        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]
            targets = targets[indices]
            pred_labels = pred_labels[indices]
        
        # Create all visualizations
        create_dimension_reduction_plots(features, targets, pred_labels, centroids, dataset_name)
        create_cluster_analysis_plots(features, targets, pred_labels, centroids, dataset_name)
        create_quality_metrics_plots(features, targets, pred_labels, dataset_name)
        create_dataset_specific_plots(features, targets, pred_labels, dataset_name)
        
    except Exception as e:
        print(f"Error in {dataset_name} visualization: {str(e)}")

if __name__ == "__main__":
    save_dir = get_save_directory()
    print(f"Using directory for saving plots: {save_dir}")
    
    for dataset in ['covertype', 'mnist', 'creditcard']:
        print(f"\nGenerating visualizations for {dataset}...")
        visualize_dataset(dataset)