"""
    Verification of Uniform Distribution by Rejection Sampling in Compositional Space using 3D Kernel 
    Density Estimation (KDE).
    
    This script demonstrates and validates that rejection sampling produces a uniform distribution
    of compositions within a constrained compositional space. Using a 3-component system (Ta-Cr-Nb)
    as an example, the script:
    
    1. Defines compositional constraints for a ternary alloy system
    2. Implements rejection sampling to generate points within the feasible region
    3. Applies Kernel Density Estimation (KDE) to visualize the distribution density
    4. Confirms that the resulting distribution is approximately uniform across the valid composition space
    
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
    Consider Ta, Cr, Nb with total constraint 100% and each element's constraint is 0-50%.
    In this case, the feasible compositional space is a triangle within a 3D space.
"""
elem_names = ['Ta', 'Cr', 'Nb']
total_constraint = 100.
comp_constraints = np.array([
    [0, 0.5],   # Ta
    [0, 0.5],   # Cr
    [0, 0.5],   # Nb
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
    Part 3: 3D KDE
    Plot 3D KDE of rejection-sampled compositions.
"""
kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(rejection_sampling_comps_df[['Ta', 'Cr', 'Nb']])

# Generate grid points for inspection
x = np.linspace(*(comp_constraints[0, :]), 51)  # Ta
y = np.linspace(*(comp_constraints[1, :]), 51)  # Cr

mesh_x, mesh_y = np.meshgrid(x, y)
mesh_x = mesh_x.ravel()
mesh_y = mesh_y.ravel()

mesh_z = total_constraint - mesh_x - mesh_y  # Nb, sum of Ta, Cr, Nb is 100%

positions = np.c_[mesh_x, mesh_y, mesh_z]
mask = (mesh_z >= comp_constraints[2, 0]) & (mesh_z <= comp_constraints[2, 1])  # Nb should be in the range of 0-50%
positions = positions[mask]

# Calculate densities at the grid points
densities = np.exp(kde.score_samples(positions)) * 1e4  # scale by 1e4

# Plot 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a normalization to map density values to colors
norm = Normalize(vmin=densities.min(), vmax=densities.max())

# Semi-transparent scatter plot, color mapped by density
scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                    c=densities, cmap='viridis', alpha=0.5, 
                    norm=norm)

# Add axis labels
ax.set_xlabel('Ta (%)', fontsize=12)
ax.set_ylabel('Cr (%)', fontsize=12)
ax.set_zlabel('Nb (%)', fontsize=12)

ax.set_xlim(0, 50)
ax.set_ylim(0, 50)
ax.set_zlim(0, 50)

# Add colorbar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label(r'Probability Density ($\times 10^{-4}$)', fontsize=14)  # density scaled by 1e4
# Format the colorbar
cbar.ax.set_position([0.72, 0.1, 0.02, 0.8])
import matplotlib.ticker as ticker
cbar.formatter = ticker.ScalarFormatter(useMathText=True)  # Use scientific notation
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3e'))  # Show 3 decimal places in scientific notation
cbar.ax.tick_params(labelsize=12, pad=8)
cbar.update_ticks()

plt.title('Kernel Density Estimation of Rejection-Sampled Compositions', fontsize=14)
plt.gca().view_init(elev=20, azim=45)
plt.savefig('3d_kde_with_colorbar.png', dpi=300)
plt.show()