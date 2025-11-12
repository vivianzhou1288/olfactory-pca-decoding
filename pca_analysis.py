"""
PCA Analysis Script for Olfactory Discrimination Neural Data
Analyzes simulated neural features (theta, beta, noise) using LFPPCAAnalyzer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pca import LFPPCAAnalyzer
from simulation import create_feature_matrix

def convert_to_lfp_format(feature_matrix, use_all_features=False):
    """
    Convert feature matrix to LFP format for LFPPCAAnalyzer.
    
    Parameters:
    -----------
    feature_matrix : pandas DataFrame
        Output from create_feature_matrix()
    use_all_features : bool
        If True, use all features. If False, use only neural features.
    
    Returns:
    --------
    lfp_data : ndarray, shape (n_trials, n_channels, n_timepoints)
    labels : ndarray
        Learning stage labels
    feature_names : list
        Names of features used as "channels"
    """
    n_trials = len(feature_matrix)
    
    # Select features to use
    if use_all_features:
        # All features
        feature_cols = [
            'OdorOn', 'Hit', 'Miss', 'CR', 'FA', 'S+', 'S-', 'Reinf',
            'theta_frequency', 'beta_frequency', 'noise_level'
        ]
    else:
        # Neural features only
        feature_cols = [
            'theta_frequency', 'beta_frequency', 'noise_level'
        ]
    
    n_channels = len(feature_cols)
    n_timepoints = 100  # Can change
    
    # Create LFP-like array
    lfp_data = np.zeros((n_trials, n_channels, n_timepoints))
    
    # Fill with feature values
    for trial_idx in range(n_trials):
        for channel_idx, feature_name in enumerate(feature_cols):
            value = feature_matrix.iloc[trial_idx][feature_name]
            lfp_data[trial_idx, channel_idx, :] = value
    
    # Get labels for visualization
    labels = feature_matrix['Learning_Stage'].values
    
    return lfp_data, labels, feature_cols


def run_pca_analysis(feature_matrix, use_all_features=False):
    """
    Run PCA using the LFPPCAAnalyzer class on simulated data.
    
    Parameters:
    -----------
    feature_matrix : pandas DataFrame
        Output from create_feature_matrix()
    use_all_features : bool
        If True, use all features. If False, use only neural features (recommended).
    
    Returns:
    --------
    analyzer : LFPPCAAnalyzer
        Fitted PCA analyzer object
    transformed : ndarray
        Data in PC space
    labels : ndarray
        Learning stage labels
    """

    # Convert to LFP format
    lfp_data, labels, feature_names = convert_to_lfp_format(
        feature_matrix, 
        use_all_features=use_all_features
    )

    analyzer = LFPPCAAnalyzer(n_components=None)

    # Extract features and fit into PCA
    features = analyzer.extract_features(lfp_data)
    transformed = analyzer.fit_transform(features, labels)

    # Print variance explained
    variance_top3 = np.sum(analyzer.pca.explained_variance_ratio_[:3]) * 100
    print(f"   Top 3 PCs explain: {variance_top3:.1f}% of variance")
    print(f"   PC1: {analyzer.pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"   PC2: {analyzer.pca.explained_variance_ratio_[1]*100:.1f}%")
    print(f"   PC3: {analyzer.pca.explained_variance_ratio_[2]*100:.1f}%")

    # Generate visualizations
    # Explained variance
    fig1 = analyzer.plot_explained_variance(n_components_to_show=10)
    plt.savefig('pca_variance.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # PCA space - Learning stage
    fig2 = analyzer.plot_pca_space(labels, pc_x=0, pc_y=1)
    plt.savefig('pca_space_learning.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # PCA space - Odorant type
    odorant_labels = feature_matrix['S+'].map({1: 'S+ (Rewarded)', 0: 'S- (Unrewarded)'}).values
    fig3 = analyzer.plot_pca_space(odorant_labels, pc_x=0, pc_y=1)
    plt.savefig('pca_space_odorant.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # PCA space - Mouse ID
    mouse_labels = feature_matrix['Mouse_ID'].values
    fig4 = analyzer.plot_pca_space(mouse_labels, pc_x=0, pc_y=1)
    plt.savefig('pca_space_mouse.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)

    # Extract loadings
    n_comps = min(3, len(analyzer.pca.explained_variance_ratio_))
    loadings = analyzer.get_loading_weights(n_components=n_comps)

    plt.show()
    return analyzer, transformed, labels

if __name__ == "__main__":
    
    # Generate simulated data
    feature_matrix, trial_labels = create_feature_matrix(n_mice=5, n_trials_per_mouse=40)
    
    print(f"\nGenerated feature matrix: {feature_matrix.shape}")
    print(f"Features: {feature_matrix.columns.tolist()}")
    
    # Run PCA on neural features only    
    analyzer, transformed_data, labels = run_pca_analysis(
        feature_matrix, 
        use_all_features=False
    )