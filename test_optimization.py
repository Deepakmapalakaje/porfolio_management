
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

# Copying optimize_weights from rebalanced_predictive_model.py to avoid import issues and dependencies
def optimize_weights(returns_df):
    """
    Hierarchical Risk Parity (HRP) Optimizer.
    Allocates weights based on hierarchical clustering and recursive bisection.
    Matches the optimization strategy from analysis.py
    """
    print("\n=== WEIGHT OPTIMIZATION PROCESS ===")

    # 1. Compute Covariance and Correlation
    cov = returns_df.cov()
    corr = returns_df.corr()

    # Handle constant returns (zero variance) to avoid NaNs
    if cov.empty or cov.values.sum() == 0:
        print("Covariance empty or zero sum")
        return np.full(returns_df.shape[1], 1.0 / returns_df.shape[1])

    print(f"Asset count: {len(returns_df.columns)}")
    # print(f"Assets: {list(returns_df.columns)}")

    # 2. Hierarchical Clustering
    # Distance matrix based on correlation
    dist = np.sqrt((1 - corr) / 2)

    # print(f"\nCorrelation Matrix:")
    # print(corr.round(3))

    try:
        # Ensure diagonal is 0 for squareform checks and fill NaNs
        np.fill_diagonal(dist.values, 0)
        dist = dist.fillna(0)

        # Convert to condensed distance matrix
        condensed_dist = squareform(dist.values)

        # Single Linkage Clustering
        link = linkage(condensed_dist, method='single')

        # print(f"\nDistance Matrix for Clustering:")
        # print(dist.round(3))

    except Exception as e:
        # Fallback if clustering fails
        print(f"Clustering failed: {e}")
        return np.full(returns_df.shape[1], 1.0 / returns_df.shape[1])

    # 3. Quasi-Diagonalization (Sort Indices)
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
    # print(f"\nQuasi-Diagonalization (Reordered Indices): {sort_ix}")
    # print(f"Reordered assets: {[returns_df.columns[i] for i in sort_ix]}")

    # 4. Recursive Bisection
    def get_cluster_var(cov, c_items):
        cov_slice = cov.iloc[c_items, c_items]
        # Inverse variance weights
        diag = np.diag(cov_slice)
        if np.any(diag == 0):
            print("Zero variance detected in diagonal")
            return 0
        w = 1 / diag
        w /= w.sum()
        # Cluster variance
        cluster_var = np.dot(np.dot(w, cov_slice), w)
        # print(f"  Cluster {c_items} variance: {cluster_var:.6f}")
        return cluster_var

    def get_rec_bipart(cov, sort_ix):
        # print(f"\n--- RECURSIVE BISECTION START ---")
        # print(f"Initial equal weights for all assets")
        w = pd.Series(1.0, index=sort_ix)
        # print(f"Current weights: {w.values}")

        c_items = [sort_ix]
        level = 0

        while len(c_items) > 0:
            level += 1
            # print(f"\nLEVEL {level}:")
            # print(f"Clusters to process: {len(c_items)}")

            new_c_items = []
            for i in range(0, len(c_items), 2):
                if i + 1 < len(c_items):
                    c_items0 = c_items[i]
                    c_items1 = c_items[i + 1]

                    # print(f"  Splitting cluster {c_items0} vs {c_items1}")
                    
                    c_var0 = get_cluster_var(cov, c_items0)
                    c_var1 = get_cluster_var(cov, c_items1)

                    if c_var0 + c_var1 == 0:
                        alpha = 0.5
                    else:
                        # Risk parity allocation
                        alpha = 1 - c_var0 / (c_var0 + c_var1)
                    
                    # print(f"    Alpha allocation: A={alpha:.3f}, B={1-alpha:.3f}")

                    # Apply weights
                    w[c_items0] *= alpha
                    w[c_items1] *= 1 - alpha

                    # print(f"    Updated weights: {w.values}")
                else:
                    # Odd number of clusters - keep the last one as is
                    new_c_items.append(c_items[i])

            # Add all non-processed clusters back
            # This logic seems slightly suspect in the original code, let's trace it carefully
            # Original: c_items = [i[j:k] for i in new_c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            # Wait, new_c_items only contains the ODD one out. The processed ones are gone?
            # NO! The processed ones are NOT added to new_c_items.
            # So where do the children of the processed clusters go?
            # They are NOT added to c_items for the next iteration?
            
            # Let's look at the original code's loop logic again.
            # for i in range(0, len(c_items), 2):
            #    if i + 1 < len(c_items):
            #       ... process pair ...
            #       # The children are NOT added to new_c_items here.
            #    else:
            #       new_c_items.append(c_items[i])
            
            # Then:
            # c_items = [i[j:k] for i in new_c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            
            # This implies that ONLY the odd one out gets split in the next step?
            # That would mean the main clusters are NEVER split further!
            # THIS LOOKS LIKE A BUG in the original code.
            
            pass 
            # I will run this as is to see if it produces equal weights.
            
            # REPRODUCING ORIGINAL LOGIC EXACTLY
            processed_children = [] 
            # Wait, the original code:
            # c_items = [i[j:k] for i in new_c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            # If new_c_items only has the leftover, then the ones that were processed (split) are dropped from c_items?
            # If they are dropped, they are never split again.
            # But their weights were updated once.
            # So they get 1 split and that's it?
            
            # Let's trace:
            # Start: [ [0, 1, 2, 3, 4] ] (len 1)
            # Loop:
            #   i=0. len=1. i+1 < len is False.
            #   Else: new_c_items.append([0, 1, 2, 3, 4])
            # c_items becomes split of [0, 1, 2, 3, 4]:
            #   [ [0, 1], [2, 3, 4] ]
            
            # Next Loop:
            #   i=0. len=2. i+1 < len is True.
            #   Process [0, 1] and [2, 3, 4].
            #   Update weights for [0, 1] and [2, 3, 4].
            #   new_c_items is EMPTY.
            # c_items becomes empty list []
            # Loop terminates.
            
            # Result: Only 1 split happens!
            # Weights are updated once based on the top-level split.
            # Inside [0, 1], the weights are never differentiated.
            # Inside [2, 3, 4], the weights are never differentiated.
            
            # So if we have 5 assets.
            # Split 1: [0, 1] vs [2, 3, 4].
            # Weights: [alpha, alpha, 1-alpha, 1-alpha, 1-alpha] (roughly)
            # They won't be equal, but they will be block-equal.
            
            # But if the user sees EXACTLY 0.2000 for all, it means alpha was 0.5?
            # Or something else.
            
            # Let's verify this logic with the script.
            
            # To fix the logic (if it is indeed broken), we should collect the children of the processed clusters too.
            
            # But first, let's run the exact copy of the code.
            
            # Re-implementing the loop exactly as in the file:
            new_c_items_original_logic = []
            for i in range(0, len(c_items), 2):
                if i + 1 < len(c_items):
                    c_items0 = c_items[i]
                    c_items1 = c_items[i + 1]
                    
                    c_var0 = get_cluster_var(cov, c_items0)
                    c_var1 = get_cluster_var(cov, c_items1)
                    
                    if c_var0 + c_var1 == 0:
                        alpha = 0.5
                    else:
                        alpha = 1 - c_var0 / (c_var0 + c_var1)
                    
                    w[c_items0] *= alpha
                    w[c_items1] *= 1 - alpha
                    
                    # The original code DOES NOT add these to new_c_items
                else:
                    new_c_items_original_logic.append(c_items[i])
            
            # Original update line:
            c_items = [i[j:k] for i in new_c_items_original_logic for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            
        return w

    # Calculate weights
    weights_series = get_rec_bipart(cov, sort_ix)

    # Return weights in original column order
    final_weights = weights_series.sort_index().values
    return final_weights

# Generate random data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=120)
data = np.random.randn(120, 5) + 0.001 # Mean return slightly positive
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'], index=dates)

print("Running optimization with random data...")
weights = optimize_weights(df)
print("\nFinal Weights:")
print(weights)

# Check if they are equal
if np.allclose(weights, 0.2):
    print("\nWeights are EQUAL (0.2). Logic might be flawed or data has no structure.")
else:
    print("\nWeights are NOT equal.")
