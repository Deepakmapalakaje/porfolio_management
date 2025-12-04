import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

# Correlation Matrix
correlation_matrix = np.array([
    [1.00, 0.21, 0.22, 0.14, 0.30],  # BAJFINANCE
    [0.21, 1.00, 0.35, 0.11, 0.25],  # SUNPHARMA
    [0.22, 0.35, 1.00, 0.21, 0.25],  # BIOCON
    [0.14, 0.11, 0.21, 1.00, 0.13],  # TI
    [0.30, 0.25, 0.25, 0.13, 1.00]   # BHARTIARTL
])

stocks = ['BAJFINANCE', 'SUNPHARMA', 'BIOCON', 'TI', 'BHARTIARTL']

print("="*60)
print("HIERARCHICAL RISK PARITY (HRP) CLUSTERING WEIGHTS")
print("="*60)
print()

# Convert correlation to distance
distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
np.fill_diagonal(distance_matrix, 0)

# Hierarchical clustering
condensed_dist = squareform(distance_matrix, checks=False)
link = linkage(condensed_dist, method='single')

# Get quasi-diagonal order
def get_quasi_diag(link):
    n = link.shape[0] + 1
    def seriation(Z, N, cur_index):
        if cur_index < N:
            return [cur_index]
        else:
            left = int(Z[cur_index - N, 0])
            right = int(Z[cur_index - N, 1])
            return seriation(Z, N, left) + seriation(Z, N, right)
    return seriation(link, n, 2 * n - 2)

quasi_diag_order = get_quasi_diag(link)

# Calculate HRP weights
def get_cluster_var(cov, items):
    cov_slice = cov[np.ix_(items, items)]
    w = np.ones(len(items)) / len(items)
    return np.dot(w, np.dot(cov_slice, w))

def get_hrp_weights(cov, sort_ix):
    w = pd.Series(1.0, index=sort_ix)
    cluster_items = [sort_ix]
    
    while len(cluster_items) > 0:
        cluster_items = [i[int(j):int(k)] for i in cluster_items 
                        for j, k in ((0, len(i) / 2), (len(i) / 2, len(i))) 
                        if len(i) > 1]
        
        for i in range(0, len(cluster_items), 2):
            c0, c1 = cluster_items[i], cluster_items[i + 1]
            v0, v1 = get_cluster_var(cov, c0), get_cluster_var(cov, c1)
            alpha = 1 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1 - alpha
    
    return w

# Use correlation as covariance (assuming unit variance)
hrp_weights = get_hrp_weights(correlation_matrix, quasi_diag_order)

# Reorder to match original stock order
final_weights = pd.Series(index=stocks, dtype=float)
for i, stock_idx in enumerate(quasi_diag_order):
    final_weights[stocks[stock_idx]] = hrp_weights.iloc[i]

# Display results
print("CLUSTERING WEIGHTS:")
print("-"*60)
for stock in stocks:
    pct = final_weights[stock] * 100
    bar = '*' * int(pct * 0.5)
    print(f"{stock:12s}: {final_weights[stock]:.6f} ({pct:5.2f}%) {bar}")
print("-"*60)
print(f"{'TOTAL':12s}: {final_weights.sum():.6f} ({final_weights.sum()*100:.2f}%)")
print("="*60)
print()

print("CLUSTERED ORDER:")
print("-"*60)
for i, idx in enumerate(quasi_diag_order):
    print(f"{i+1}. {stocks[idx]}")
print("="*60)
print()

print("KEY INSIGHTS:")
print("-"*60)
print("• Stocks with higher correlations are clustered together")
print("• Weights are allocated inversely to cluster variance")
print("• This provides better diversification than equal weighting")
print("="*60)
