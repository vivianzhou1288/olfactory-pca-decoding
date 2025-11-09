import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import signal
import pandas as pd

class LFPPCAAnalyzer:
    """
    Perform PCA on Local Field Potential (LFP) data from neural recordings.
    Designed for multi-electrode recordings during behavioral experiments.
    """
    
    def __init__(self, n_components=None):
        """
        Initialize PCA analyzer.
        
        Parameters:
        -----------
        n_components : int or None
            Number of principal components to keep. 
            If None, keep all components.
        """
        self.n_components = n_components
        self.pca = None
        self.scaler = StandardScaler()
        self.transformed_data = None
        
    def preprocess_lfp(self, lfp_data, fs, lowcut=1, highcut=100):
        """
        Preprocess LFP data with bandpass filtering.
        
        Parameters:
        -----------
        lfp_data : ndarray, shape (n_channels, n_timepoints) or (n_trials, n_channels, n_timepoints)
            Raw LFP recordings
        fs : float
            Sampling frequency in Hz
        lowcut : float
            Low cutoff frequency for bandpass filter
        highcut : float
            High cutoff frequency for bandpass filter
            
        Returns:
        --------
        filtered_data : ndarray
            Bandpass filtered LFP data
        """
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        if lfp_data.ndim == 2:
            # Single trial: (n_channels, n_timepoints)
            filtered_data = signal.filtfilt(b, a, lfp_data, axis=1)
        elif lfp_data.ndim == 3:
            # Multiple trials: (n_trials, n_channels, n_timepoints)
            filtered_data = np.zeros_like(lfp_data)
            for trial in range(lfp_data.shape[0]):
                filtered_data[trial] = signal.filtfilt(b, a, lfp_data[trial], axis=1)
        else:
            filtered_data = signal.filtfilt(b, a, lfp_data)
            
        return filtered_data
    
    def extract_features(self, lfp_data, time_window=None):
        """
        Extract features from LFP data for PCA.
        
        Parameters:
        -----------
        lfp_data : ndarray, shape (n_trials, n_channels, n_timepoints)
            Preprocessed LFP data
        time_window : tuple or None
            (start_idx, end_idx) to extract specific time window
            
        Returns:
        --------
        features : ndarray, shape (n_trials, n_features)
            Feature matrix where each row is a trial
        """
        if time_window is not None:
            start_idx, end_idx = time_window
            lfp_data = lfp_data[:, :, start_idx:end_idx]
        
        n_trials = lfp_data.shape[0]
        
        # Flatten each trial: (n_channels * n_timepoints) features per trial
        features = lfp_data.reshape(n_trials, -1)
        
        return features
    
    def fit_transform(self, features, labels=None):
        """
        Fit PCA model and transform data.
        
        Parameters:
        -----------
        features : ndarray, shape (n_samples, n_features)
            Feature matrix
        labels : array-like, optional
            Labels for each sample (e.g., trial type, condition)
            
        Returns:
        --------
        transformed : ndarray, shape (n_samples, n_components)
            Transformed data in PC space
        """
        # Standardize features (zero mean, unit variance)
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.transformed_data = self.pca.fit_transform(features_scaled)
        
        return self.transformed_data
    
    def plot_explained_variance(self, n_components_to_show=20):
        """Plot cumulative explained variance."""
        if self.pca is None:
            raise ValueError("PCA not fitted yet. Call fit_transform first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Individual explained variance
        n_show = min(n_components_to_show, len(self.pca.explained_variance_ratio_))
        ax1.bar(range(1, n_show + 1), 
                self.pca.explained_variance_ratio_[:n_show] * 100)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance (%)')
        ax1.set_title('Variance Explained by Each PC')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative explained variance
        cumsum = np.cumsum(self.pca.explained_variance_ratio_) * 100
        ax2.plot(range(1, len(cumsum) + 1), cumsum, 'o-')
        ax2.axhline(y=95, color='r', linestyle='--', label='95% variance')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance (%)')
        ax2.set_title('Cumulative Variance Explained')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_pca_space(self, labels=None, pc_x=0, pc_y=1):
        """
        Plot data in 2D principal component space.
        
        Parameters:
        -----------
        labels : array-like, optional
            Labels for coloring points (e.g., trial conditions)
        pc_x, pc_y : int
            Which principal components to plot (0-indexed)
        """
        if self.transformed_data is None:
            raise ValueError("No transformed data. Call fit_transform first.")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(self.transformed_data[mask, pc_x],
                          self.transformed_data[mask, pc_y],
                          c=[colors[i]], label=str(label), alpha=0.6, s=50)
            ax.legend()
        else:
            ax.scatter(self.transformed_data[:, pc_x],
                      self.transformed_data[:, pc_y],
                      alpha=0.6, s=50)
        
        var_x = self.pca.explained_variance_ratio_[pc_x] * 100
        var_y = self.pca.explained_variance_ratio_[pc_y] * 100
        
        ax.set_xlabel(f'PC{pc_x+1} ({var_x:.1f}% variance)')
        ax.set_ylabel(f'PC{pc_y+1} ({var_y:.1f}% variance)')
        ax.set_title('LFP Data in Principal Component Space')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_loading_weights(self, n_components=3):
        """
        Get the loading weights (contributions) for top PCs.
        
        Parameters:
        -----------
        n_components : int
            Number of top components to return
            
        Returns:
        --------
        loadings : ndarray, shape (n_components, n_features)
            Loading weights for each PC
        """
        if self.pca is None:
            raise ValueError("PCA not fitted yet.")
        
        return self.pca.components_[:n_components]


# Example usage
def example_analysis():
    """
    LFPPCAAnalyzer with simulated data.
    """
    # Simulate some LFP data
    n_trials = 100
    n_channels = 16
    n_timepoints = 1000
    fs = 1000  # Sampling frequency in Hz
    
    # Simulate two conditions (e.g., Hit vs Miss trials)
    condition1_trials = 50
    condition2_trials = 50
    
    # Generate synthetic data (replace with your actual data loading)
    np.random.seed(42)
    lfp_data = np.random.randn(n_trials, n_channels, n_timepoints)
    
    # Add some structure to differentiate conditions
    lfp_data[:condition1_trials] += np.sin(2 * np.pi * 10 * 
                                            np.linspace(0, 1, n_timepoints))
    lfp_data[condition1_trials:] += np.cos(2 * np.pi * 15 * 
                                            np.linspace(0, 1, n_timepoints))
    
    labels = np.array(['Condition_1'] * condition1_trials + 
                     ['Condition_2'] * condition2_trials)
    
    # Initialize analyzer
    analyzer = LFPPCAAnalyzer(n_components=10)
    
    # Preprocess
    print("Preprocessing LFP data...")
    filtered_data = analyzer.preprocess_lfp(lfp_data, fs)
    
    # Extract features
    print("Extracting features...")
    features = analyzer.extract_features(filtered_data)
    print(f"Feature matrix shape: {features.shape}")
    
    # Fit PCA
    print("Fitting PCA...")
    transformed = analyzer.fit_transform(features, labels)
    print(f"Transformed data shape: {transformed.shape}")
    
    # Visualize results
    print("\nGenerating plots...")
    analyzer.plot_explained_variance()
    analyzer.plot_pca_space(labels, pc_x=0, pc_y=1)
    plt.show()
    
    # Print variance explained
    print(f"\nTop 5 PCs explain: "
          f"{np.sum(analyzer.pca.explained_variance_ratio_[:5])*100:.1f}% of variance")

if __name__ == "__main__":
    example_analysis()