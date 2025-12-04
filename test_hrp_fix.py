import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

# Test the HRP algorithm with your correlation matrix
correlation_matrix = np.array([
    [1.00, 0.21, 0.22, 0.14, 0.30],
    [0.21, 1.00, 0.35, 0.11, 0.25],
    [0.22, 0.35, 1.00, 0.21, 0.25],
    [0.14, 0.11, 0.21, 1.00, 0.13],
    [0.30, 0.25, 0.25, 0.13, 1.00]
])

stocks = ['BAJFINANCE', 'SUNPHARMA', 'BIOCON', 'TI', 'BHARTIARTL']

# Create a DataFrame with sample returns (using correlation as covariance for testing)
returns_df = pd.DataFrame(correlation_matrix, columns=stocks)

# Simulate the optimize_weights function
def optimize_weights_test(returns_df):
    """Test HRP Optimizer"""
    print("=== WEIGHT OPTIMIZATION TEST ===\n")
    
    # 1. Compute Covariance and Correlation
    cov = returns_df.cov()
    corr = returns_df.corr()
    
    print(f"Asset count: {len(returns_df.columns)}")
    print(f"Assets: {list(returns_df.columns)}\n")
    
    # 2. Hierarchical Clustering
    dist = np.sqrt((1 - corr) / 2)
    
    print("Correlation Matrix:")
    print(corr.round(3))
    print()
    
    np.fill_diagonal(dist.values, 0)
    dist = dist.fillna(0)
    
    condensed_dist = squareform(dist.values)
    link = linkage(condensed_dist, method='single')
    
    print("Distance Matrix:")
    print(dist.round(3))
    print()
    
    # 3. Quasi-Diagonalization
    def get_quasi_diag(link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()
    
    sort_ix = get_quasi_diag(link)
    print(f"Quasi-Diagonalization: {sort_ix}")
    print(f"Reordered assets: {[returns_df.columns[i] for i in sort_ix]}\n")
    
    # 4. Recursive Bisection
    def get_cluster_var(cov, c_items):
        cov_slice = cov.iloc[c_items, c_items]
        w = 1 / np.diag(cov_slice)
        w /= w.sum()
        cluster_var = np.dot(np.dot(w, cov_slice), w)
        print(f"  Cluster {c_items} variance: {cluster_var:.6f}")
        return cluster_var
    
    def get_rec_bipart(cov, sort_ix):
        print("--- RECURSIVE BISECTION START ---")
        w = pd.Series(1.0, index=sort_ix)
        print(f"Initial weights: {w.values}\n")
        
        c_items = [sort_ix]
        level = 0
        
        while len(c_items) > 0:
            level += 1
            print(f"LEVEL {level}:")
            print(f"Clusters to process: {len(c_items)}")
            
            new_c_items = []
            for cluster in c_items:
                if len(cluster) > 1:
                    mid = len(cluster) // 2
                    c_items0 = cluster[:mid]
                    c_items1 = cluster[mid:]
                    
                    print(f"\n  Splitting cluster {cluster}")
                    print(f"    Cluster A: {[returns_df.columns[j] for j in c_items0]}")
                    print(f"    Cluster B: {[returns_df.columns[j] for j in c_items1]}")
                    
                    c_var0 = get_cluster_var(cov, c_items0)
                    c_var1 = get_cluster_var(cov, c_items1)
                    
                    alpha = 1 - c_var0 / (c_var0 + c_var1)
                    print(f"    Alpha: A={alpha:.3f}, B={1-alpha:.3f}")
                    
                    w[c_items0] *= alpha
                    w[c_items1] *= 1 - alpha
                    
                    print(f"    Weights after split: {w.values}")
                    
                    new_c_items.append(c_items0)
                    new_c_items.append(c_items1)
            
            c_items = new_c_items
            print()
        
        print(f"FINAL WEIGHTS (sorted order): {w.values}")
        return w
    
    weights_series = get_rec_bipart(cov, sort_ix)
    final_weights = weights_series.sort_index().values
    
    print("\n" + "="*60)
    print("FINAL WEIGHTS (original order):")
    print("="*60)
    for i, asset in enumerate(returns_df.columns):
        print(f"  {asset:12s}: {final_weights[i]:.6f} ({final_weights[i]*100:.2f}%)")
    print("-"*60)
    print(f"  {'TOTAL':12s}: {final_weights.sum():.6f} ({final_weights.sum()*100:.2f}%)")
    print("="*60)
    
    return final_weights

# Run the test
weights = optimize_weights_test(returns_df)

print("\n\nEXPECTED WEIGHTS (from our calculation):")
print("="*60)
expected = {
    'BAJFINANCE': 0.158745,
    'SUNPHARMA': 0.238070,
    'BIOCON': 0.206369,
    'TI': 0.238070,
    'BHARTIARTL': 0.158745
}
for stock, weight in expected.items():
    print(f"  {stock:12s}: {weight:.6f} ({weight*100:.2f}%)")
print("="*60)
