"""
State Space Entropy Analysis for Bayesian Optimization

This module implements tools to analyze the exploration behavior of Bayesian optimization
by calculating the k-nearest neighbor entropy (state entropy) of composition samples.

State entropy provides a quantitative measure of how the optimization algorithm
explores the composition space over time. Higher entropy indicates more diverse
exploration, while lower entropy suggests more focused exploitation of specific regions.

This analysis helps evaluate the exploration-exploitation trade-off in different
Bayesian optimization strategies for materials discovery.
"""
from environment import *
from scipy.spatial.distance import pdist, squareform
from glob import glob

def cal_knn_entropy(samples, k=3):
    """
    Calculate k-nearest neighbor entropy of sample points
    
    This information-theoretic measure quantifies the diversity and coverage
    of points in a multidimensional space. It estimates local sample density
    using the distances to the k-nearest neighbors.
    
    The formula is: H_k(X) = log(mean(distance to k-th nearest neighbor))
    
    Higher entropy values indicate more dispersed sampling (exploration),
    while lower values suggest clustering around specific regions (exploitation).
    
    Args:
        samples: Array of sample points (compositions)
        k: Number of nearest neighbors to consider (default: 3)
        
    Returns:
        Entropy value based on k-nearest neighbors distances
    """
    # Calculate pairwise distances between all sample points
    dist_mat = squareform(pdist(samples, metric='euclidean'))
    
    # Sort distances for each point to find k-nearest neighbors
    dist_mat = np.sort(dist_mat)
    
    # Calculate entropy: log of mean distance to k-th nearest neighbor
    # (skipping 0th distance which is self-distance)
    return np.log(dist_mat[:, k]).mean()

# Minimum number of initial samples required before entropy calculation
init_smpl_len = 10

def ke_traj(path: str):
    """
    Calculate entropy trajectory from a Bayesian optimization run
    
    Computes how state entropy evolves throughout the optimization process
    as more compositions are sampled.
    
    Args:
        path: Path to saved optimization results file
        
    Returns:
        List of entropy values at each step of the optimization
    """
    # Load compositions and targets from saved results
    comps, _ = joblib.load(path)
    
    # Calculate entropy at each step of the optimization
    # (starting from init_smpl_len samples)
    ke_buffer = []
    for i in range(init_smpl_len, len(comps)):
        # Calculate entropy of all compositions up to current step
        ke_buffer.append(cal_knn_entropy(comps[:i]))

    return ke_buffer

    # The following code is unreachable but left for reference
    # Print entropy values with corresponding experiment indices
    for exp_idx, ke_buffer in zip(np.arange(41, len(comps) + 1), ke_buffer):
        print(exp_idx, ke_buffer)

def ke_analyse(path_reg: str):
    """
    Analyze entropy trajectories across multiple optimization runs
    
    Computes statistical metrics (mean and standard deviation) of entropy
    trajectories from multiple Bayesian optimization runs to evaluate
    the consistency of exploration behavior.
    
    Args:
        path_reg: Glob pattern to match multiple result files
    """
    # Collect paths to all optimization result files
    paths = glob(path_reg)
    
    # Calculate entropy trajectories for each optimization run
    ke_trajs = [ke_traj(p) for p in paths]
    ke_trajs = np.array(ke_trajs)

    # Define output path in the same directory as input files
    save_path = os.path.join(os.path.dirname(path_reg), '~state_entropy_res.txt')
    
    # Save results: experiment index, mean entropy, standard deviation
    np.savetxt(
        save_path, 
        np.vstack((
            # Experiment indices (adding init_smpl_len offset)
            np.arange(len(ke_trajs.T)) + init_smpl_len + 1,
            # Mean entropy across runs at each step
            ke_trajs.mean(axis=0),
            # Standard deviation of entropy across runs
            ke_trajs.std(axis=0),
        )).T, 
        delimiter='\t'
    )

# Run entropy analysis on optimization results 
# (from state_entropy directory)
ke_analyse('results\\state_entropy\\*pkl')