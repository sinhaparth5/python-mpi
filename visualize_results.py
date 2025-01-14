# visualize_results.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

# Try results directory first, fallback to result_img
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
            # Use current directory as last resort
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
            # Final fallback - save in current directory
            plt.savefig(filename)
            print(f"Saved in current directory: {filename}")
        except Exception as e:
            print(f"Failed to save plot anywhere: {e}")

def visualize_iris():
    """Visualize Iris dataset clustering results"""
    try:
        # Load original data
        iris_data = pd.read_csv('data/iris_data.csv')
        X = iris_data.drop('target', axis=1).values
        true_labels = iris_data['target'].values
        
        # Load clustering results
        centroids = np.load(os.path.join(results_dir, 'iris_centroids.npy'))
        pred_labels = np.load(os.path.join(results_dir, 'iris_labels.npy'))
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot using first two features
        scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis')
        ax1.set_title('Iris Original Labels')
        ax1.set_xlabel('Sepal Length')
        ax1.set_ylabel('Sepal Width')
        plt.colorbar(scatter1, ax=ax1)
        
        scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=pred_labels, cmap='viridis')
        ax2.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
        ax2.set_title('Iris Clustering Results')
        ax2.set_xlabel('Sepal Length')
        ax2.set_ylabel('Sepal Width')
        plt.colorbar(scatter2, ax=ax2)
        
        plt.tight_layout()
        save_plot(plt, 'iris_visualization.png')
        plt.close()
        
    except Exception as e:
        print(f"Error in iris visualization: {e}")

def visualize_wine():
    """Visualize Wine dataset clustering results using PCA"""
    try:
        # Load original data
        wine_data = pd.read_csv('data/wine_data.csv')
        X = wine_data.drop('target', axis=1).values
        true_labels = wine_data['target'].values
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Load clustering results
        centroids = np.load(os.path.join(results_dir, 'wine_centroids.npy'))
        pred_labels = np.load(os.path.join(results_dir, 'wine_labels.npy'))
        centroids_pca = pca.transform(centroids)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap='viridis')
        ax1.set_title('Wine Original Labels (PCA)')
        ax1.set_xlabel('First Principal Component')
        ax1.set_ylabel('Second Principal Component')
        plt.colorbar(scatter1, ax=ax1)
        
        scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=pred_labels, cmap='viridis')
        ax2.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
        ax2.set_title('Wine Clustering Results (PCA)')
        ax2.set_xlabel('First Principal Component')
        ax2.set_ylabel('Second Principal Component')
        plt.colorbar(scatter2, ax=ax2)
        
        plt.tight_layout()
        save_plot(plt, 'wine_visualization.png')
        plt.close()
        
    except Exception as e:
        print(f"Error in wine visualization: {e}")

if __name__ == "__main__":
    save_dir = get_save_directory()
    print(f"Using directory for saving plots: {save_dir}")
    
    # Generate visualizations
    print("\nGenerating Iris visualization...")
    visualize_iris()
    
    print("\nGenerating Wine visualization...")
    visualize_wine()