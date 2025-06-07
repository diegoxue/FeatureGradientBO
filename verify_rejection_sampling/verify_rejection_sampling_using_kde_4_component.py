"""
    Verification of Uniform Distribution by Rejection Sampling in Compositional Space using Kernel 
    Density Estimation (KDE) and chi-square test.
    
    This script demonstrates and validates that rejection sampling produces a uniform distribution
    of compositions within a constrained compositional space. Using a 4-component system (Ta-Cr-Nb-Mn)
    as an example, the script:
    
    1. Defines compositional constraints for a quaternary alloy system
    2. Implements rejection sampling to generate points within the feasible region
    3. Applies Kernel Density Estimation (KDE) to evaluate the distribution density
    4. Performs chi-square test to statistically verify the uniformity of the distribution
    
    author: Yuehui Xian
    date: 2025-06-05
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

"""
    Part 1: Compositional Constraints
    Consider Ta, Cr, Nb, Mn with total constraint 100% and each element's constraint is 0-50%.
    In this case, the feasible compositional space is a tetrahedron within a 4D space.
"""
elem_names = ['Ta', 'Cr', 'Nb', 'Mn']
total_constraint = 100.
comp_constraints = np.array([
    [0, 0.5],   # Ta
    [0, 0.5],   # Cr
    [0, 0.5],   # Nb
    [0, 0.5],   # Mn
]) * total_constraint

"""
    Part 2: Rejection Sampling
    Generate 100000 samples using rejection sampling.
"""
def rejection_sampling(num_samples):
    comps_buffer = []
    while len(comps_buffer) < num_samples:
        comp_candidate = np.random.uniform(comp_constraints[:-1, 0], comp_constraints[:-1, 1], (num_samples, len(elem_names) - 1))
        comp_candidate = np.hstack((comp_candidate, total_constraint - np.sum(comp_candidate, axis=1, keepdims=True)))
        
        mask = (comp_candidate[:, -1] >= comp_constraints[-1, 0]) & (comp_candidate[:, -1] <= comp_constraints[-1, 1])
        
        comp_candidate = comp_candidate[mask]
        comps_buffer.extend(comp_candidate.tolist()[:num_samples - len(comps_buffer)])

    return np.array(comps_buffer)

rejection_sampling_comps = rejection_sampling(num_samples = 250000)

rejection_sampling_comps_df = pd.DataFrame(rejection_sampling_comps, columns=elem_names).assign(type='rejection sampling')

"""
    Part 3: KDE Setup
    Setup Kernel Density Estimation for the 4-component system.
"""
kde_bandwidth = 2
kde = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth).fit(rejection_sampling_comps_df[elem_names])

# Generate grid points for inspection
x = np.linspace(*(comp_constraints[0, :]), 26)  # Ta
y = np.linspace(*(comp_constraints[1, :]), 26)  # Cr
z = np.linspace(*(comp_constraints[2, :]), 26)  # Nb
w = np.linspace(*(comp_constraints[3, :]), 26)  # Mn

mesh_x, mesh_y, mesh_z, mesh_w = np.meshgrid(x, y, z, w)
mesh_x = mesh_x.ravel()
mesh_y = mesh_y.ravel()
mesh_z = mesh_z.ravel()
mesh_w = mesh_w.ravel()

mesh_total = mesh_x + mesh_y + mesh_z + mesh_w  # sum of Ta, Cr, Nb, Mn is 100%

""" 
    Part 4: Statistical Analysis
    Apply KDE to grid points and perform chi-square test to verify uniformity.
    
    NOTE:   KDE bandwidth is subtracted from the constraint limits as kde 
            tends to underestimate the density at the boundary.
"""
positions = np.c_[mesh_x, mesh_y, mesh_z, mesh_w]
mask_1 = (mesh_x >= comp_constraints[0, 0] + kde_bandwidth) & (mesh_x <= comp_constraints[0, 1] - kde_bandwidth)
mask_2 = (mesh_y >= comp_constraints[1, 0] + kde_bandwidth) & (mesh_y <= comp_constraints[1, 1] - kde_bandwidth)
mask_3 = (mesh_z >= comp_constraints[2, 0] + kde_bandwidth) & (mesh_z <= comp_constraints[2, 1] - kde_bandwidth)
mask_4 = (mesh_w >= comp_constraints[3, 0] + kde_bandwidth) & (mesh_w <= comp_constraints[3, 1] - kde_bandwidth)
positions = positions[mask_1 & mask_2 & mask_3 & mask_4]

# Calculate densities at the grid points
densities = np.exp(kde.score_samples(positions))

""" chisquare test """
from scipy import stats
# suppose densities is the density value of kde
# for uniform distribution, the theoretical density should be the same
theoretical_density = np.ones_like(densities) * np.mean(densities)

chi2_stat, p_value = stats.chisquare(densities, f_exp=theoretical_density)

print(f"chisquare statistic: {chi2_stat}")
print(f"p-value: {p_value}")

if p_value < 0.05:
    print("Reject the uniform distribution hypothesis: There is sufficient evidence that the density distribution is not uniform (p = {:.4f}).".format(p_value))
else:
    print("Cannot reject the uniform distribution hypothesis: The deviation from uniformity may be due to random fluctuations (p = {:.4f}).".format(p_value))