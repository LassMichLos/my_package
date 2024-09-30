
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class DataVisualizer:
    def __init__(self, data: pd.DataFrame):
        """Initialize the DataVisualizer with a pandas DataFrame and optional labels."""
        self.data = data

    def basic_statistics(self):
        """Display basic statistics of the DataFrame."""
        return self.data.describe()

    def missing_values(self):
        """Show the count of missing values in each column."""
        return self.data.isnull().sum()

    def pca_2d(self):
        """Perform PCA on the numeric data and return the PCA results."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        scaled_data = StandardScaler().fit_transform(numeric_data)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        return pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

    def plot_2d(self, labels):
        """Plot data points colored by their true classification.""" 

        pca_df = self.pca_2d()
        pca_df['True Label'] = labels

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='True Label', palette='viridis', alpha=0.7)
        plt.title('True Classification of Data Points')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='True Label')
        plt.grid()
        plt.show()    
