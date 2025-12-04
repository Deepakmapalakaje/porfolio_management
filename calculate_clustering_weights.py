import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

# Correlation Matrix
correlation_matrix = np.array([
    [1.00, 0.21, 0.22, 0.14, 0.30],  # BAJFINANCE
    [0.21, 1.00, 0.35, 0.11, 0.25],  # SUNPHARMA
    [0.22, 0.35, 1.00, 0.21, 0.25],  # BIOCON
    [0.14, 0.11, 0.21, 1.00, 0.13],  # TI
    [0.30, 0.25, 0.25, 0.13, 1.00]   # BHARTIARTL
])

stocks = ['BAJFINANCE', 'SUNPHARMA', 'BIOCON', 'TI', 'BHARTIARTL']

# Create DataFrame for better visualization
corr_df = pd.DataFrame(correlation_matrix, index=stocks, columns=stocks)
print("Correlation Matrix:")
print(corr_df)
print("\n" + "="*80 + "\n")

# Step 1: Convert correlation to distance matrix
# Distance = sqrt(0.5 * (1 - correlation))
distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is 0

print("Distance Matrix:")
dist_df = pd.DataFrame(distance_matrix, index=stocks, columns=stocks)
print(dist_df)
print("\n" + "="*80 + "\n")

# Step 2: Perform hierarchical clustering
# Convert to condensed distance matrix for linkage
condensed_dist = squareform(distance_matrix, checks=False)
linkage_matrix = linkage(condensed_dist, method='single')

print("Linkage Matrix:")
print("(Each row: [cluster1, cluster2, distance, num_items])")
for i, row in enumerate(linkage_matrix):
    print(f"Step {i+1}: {row}")
print("\n" + "="*80 + "\n")

# Step 3: Get quasi-diagonalization order
def get_quasi_diag(link):
    """Get the order of assets from hierarchical clustering"""
    n = link.shape[0] + 1
    
    def seriation(Z, N, cur_index):
        """
        Recursive function to order leaves in the dendrogram
        """
        if cur_index < N:
            return [cur_index]
        else:
            left = int(Z[cur_index - N, 0])
            right = int(Z[cur_index - N, 1])
            return seriation(Z, N, left) + seriation(Z, N, right)
    
    # Start from the root (last merge)
    res = seriation(link, n, 2 * n - 2)
    return res

quasi_diag_order = get_quasi_diag(linkage_matrix)
print(f"Quasi-Diagonal Order (indices): {quasi_diag_order}")
print(f"Quasi-Diagonal Order (stocks): {[stocks[i] for i in quasi_diag_order]}")
print("\n" + "="*80 + "\n")

# Step 4: Calculate HRP weights using recursive bisection
def get_cluster_var(cov_matrix, cluster_items):
    """Calculate cluster variance"""
    cov_slice = cov_matrix[np.ix_(cluster_items, cluster_items)]
    w = np.ones(len(cluster_items)) / len(cluster_items)  # Equal weight within cluster
    cluster_var = np.dot(w, np.dot(cov_slice, w))
    return cluster_var

def get_rec_bipart(cov_matrix, sort_ix):
    """Recursive bisection to get HRP weights"""
    w = pd.Series(1.0, index=sort_ix)
    cluster_items = [sort_ix]
    
    while len(cluster_items) > 0:
        # Bisect clusters
        cluster_items = [i[int(j):int(k)] for i in cluster_items 
                        for j, k in ((0, len(i) / 2), (len(i) / 2, len(i))) 
                        if len(i) > 1]
        
        # Parse in pairs
        for i in range(0, len(cluster_items), 2):
            cluster0 = cluster_items[i]
            cluster1 = cluster_items[i + 1]
            
            # Calculate cluster variances
            var0 = get_cluster_var(cov_matrix, cluster0)
            var1 = get_cluster_var(cov_matrix, cluster1)
            
            # Allocate weight inversely proportional to variance
            alpha = 1 - var0 / (var0 + var1)
            
            # Update weights
            w[cluster0] *= alpha
            w[cluster1] *= 1 - alpha
            
            print(f"Cluster {[stocks[j] for j in cluster0]} vs {[stocks[j] for j in cluster1]}")
            print(f"  Var0: {var0:.6f}, Var1: {var1:.6f}, Alpha: {alpha:.6f}")
            print(f"  Weight allocation: {alpha:.4f} vs {1-alpha:.4f}")
    
    return w

# For HRP, we need a covariance matrix
# Since we only have correlation, we'll assume unit variance (std dev = 1)
# This means covariance = correlation
cov_matrix = correlation_matrix.copy()

print("Calculating HRP Weights using Recursive Bisection:")
print("-" * 80)
hrp_weights = get_rec_bipart(cov_matrix, quasi_diag_order)
print("\n" + "="*80 + "\n")

# Reorder weights to match original stock order
final_weights = pd.Series(index=stocks, dtype=float)
for i, stock_idx in enumerate(quasi_diag_order):
    final_weights[stocks[stock_idx]] = hrp_weights.iloc[i]

print("FINAL CLUSTERING WEIGHTS (HRP):")
print("="*80)
for stock in stocks:
    print(f"{stock:12s}: {final_weights[stock]:.6f} ({final_weights[stock]*100:.2f}%)")
print("-" * 80)
print(f"{'TOTAL':12s}: {final_weights.sum():.6f} ({final_weights.sum()*100:.2f}%)")
print("="*80)

# Visualize the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=stocks, orientation='top')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Stocks')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('clustering_dendrogram.png', dpi=300, bbox_inches='tight')
print("\nDendrogram saved as 'clustering_dendrogram.png'")

# Additional Analysis
print("\n" + "="*80)
print("ADDITIONAL ANALYSIS:")
print("="*80)

# Calculate inverse variance weights for comparison
variances = np.diag(cov_matrix)
inv_var_weights = (1 / variances) / np.sum(1 / variances)
print("\nInverse Variance Weights (for comparison):")
for i, stock in enumerate(stocks):
    print(f"{stock:12s}: {inv_var_weights[i]:.6f} ({inv_var_weights[i]*100:.2f}%)")

# Calculate equal weights for comparison
equal_weights = np.ones(len(stocks)) / len(stocks)
print("\nEqual Weights (for comparison):")
for i, stock in enumerate(stocks):
    print(f"{stock:12s}: {equal_weights[i]:.6f} ({equal_weights[i]*100:.2f}%)")

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print("""
The HRP (Hierarchical Risk Parity) weights are calculated based on:
1. Converting correlation to distance matrix
2. Hierarchical clustering using single linkage
3. Recursive bisection allocating weights inversely proportional to cluster variance

Key insights from your correlation matrix:
- SUNPHARMA and BIOCON have high correlation (0.35) - likely clustered together
- TI has low correlations with others - likely gets separate allocation
- BAJFINANCE and BHARTIARTL have moderate correlation (0.30)

The weights reflect diversification benefits by:
- Reducing allocation to highly correlated assets
- Increasing allocation to assets that provide diversification
""")
