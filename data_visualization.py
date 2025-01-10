import numpy as np
from sklearn.manifold import TSNE  # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder  # type: ignore
from data_processing import DataProcessor
from sklearn.cluster import KMeans, DBSCAN  # type: ignore

def reduce_dimensionality_with_tsne(df: pd.DataFrame) -> pd.DataFrame:
    if 'embedding' not in df.columns:
        print("[ERROR] 'embedding' column not found in the DataFrame.")
        return df
    
    embeddings = df['embedding'].tolist()
    if not all(len(embedding) == 320 for embedding in embeddings):
        print("[ERROR] Some embeddings do not have the expected 320 dimensions.")
        return df
    
    embeddings = np.array(embeddings)  # type: ignore
    
    tsne = TSNE(n_components=2, random_state=42)  # random state could be any seed
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    df['tsne_x'] = reduced_embeddings[:, 0]
    df['tsne_y'] = reduced_embeddings[:, 1]
    
    return df

def plot_tsne_with_syndrome_id(df: pd.DataFrame) -> None:
    if 'tsne_x' not in df.columns or 'tsne_y' not in df.columns:
        print("[ERROR] t-SNE components not found in the DataFrame.")
        return
    
    label_encoder = LabelEncoder()
    syndrome_id_encoded = label_encoder.fit_transform(df['syndrome_id'])
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['tsne_x'], df['tsne_y'], c=syndrome_id_encoded, cmap='viridis', s=10)
    plt.colorbar(scatter, label='Syndrome ID')
    plt.title('t-SNE Visualization of Embeddings Colored by Syndrome ID')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    plt.savefig('tsne_with_syndrome_id.png')  
    plt.close()

def plot_tsne_with_clusters(df: pd.DataFrame) -> None:
    if 'tsne_x' not in df.columns or 'tsne_y' not in df.columns:
        print("[ERROR] t-SNE components not found in the DataFrame.")
        return
    
    syndrome_id_numeric = pd.factorize(df['syndrome_id'])[0]

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['tsne_x', 'tsne_y']])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['tsne_x'], df['tsne_y'], c=df['cluster'], cmap='viridis', s=10)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title('t-SNE Visualization of Embeddings with KMeans Clustering')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # Map density using hexbin
    plt.figure(figsize=(10, 8))
    plt.hexbin(df['tsne_x'], df['tsne_y'], gridsize=50, cmap='YlGnBu')
    plt.colorbar(label='Density')
    plt.title('Density Map of t-SNE Components')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    plt.savefig('tsne_with_clusters_and_density.png')  
    plt.close()  

def create_syndrome_collage(df: pd.DataFrame, output_path: str) -> None:
    if 'tsne_x' not in df.columns or 'tsne_y' not in df.columns or 'syndrome_id' not in df.columns:
        print("[ERROR] t-SNE components or syndrome_id column missing.")
        return

    unique_syndromes = df['syndrome_id'].unique()
    n_syndromes = len(unique_syndromes)
    
    n_cols = 3  
    n_rows = (n_syndromes + n_cols - 1) // n_cols  
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, syndrome in enumerate(unique_syndromes):
        syndrome_data = df[df['syndrome_id'] == syndrome]
        ax = axes[i]
        ax.scatter(syndrome_data['tsne_x'], syndrome_data['tsne_y'], s=10)
        ax.set_title(f'Syndrome {syndrome}')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_and_create_collage(df: pd.DataFrame, output_path: str) -> None:
    df_with_tsne = reduce_dimensionality_with_tsne(df)
    
    plot_tsne_with_syndrome_id(df_with_tsne)
    plot_tsne_with_clusters(df_with_tsne)
    create_syndrome_collage(df_with_tsne, output_path)