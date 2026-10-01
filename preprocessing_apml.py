"""
Initial data exploration for the APML diffusion MRI project.

This script:
1. Loads the MRI measurements from one voxel.
2. Inspects the provided DTI point estimate.
3. Constructs and checks the diffusion tensor.
4. Connects the gradient information to x in the model.
5. Predicts the MRI signals using the point estimate.
6. Compares the predictions with the measured signals.
"""

import numpy as np
import matplotlib.pyplot as plt

from dmri_project import get_preprocessed_data, compute_D


# ============================================================
# 1. Load the data
# ============================================================

y, point_estimate, gtab = get_preprocessed_data()

print("\n--- DATASET ---")
print("Number of MRI measurements:", len(y))
print("Shape of y:", y.shape)
print("First ten measured signals:")
print(y[:10])


# ============================================================
# 2. Inspect the provided point estimate
# ============================================================

S0_init, evals_init, evecs_init = point_estimate

print("\n--- INITIAL POINT ESTIMATE ---")
print("Initial S0:")
print(S0_init)

print("\nEigenvalues:")
print(evals_init)

print("\nEigenvectors:")
print(evecs_init)


# ============================================================
# 3. Construct the diffusion tensor
#
# D = V Lambda V^T
# ============================================================

D_init = compute_D(evals_init, evecs_init).squeeze()

print("\n--- DIFFUSION TENSOR ---")
print("Initial diffusion tensor D:")
print(D_init)

print("\nShape of D:", D_init.shape)
print("Is D symmetric?", np.allclose(D_init, D_init.T))

D_eigenvalues = np.linalg.eigvalsh(D_init)

print("Eigenvalues calculated from D:")
print(D_eigenvalues)

print("Are all eigenvalues positive?", np.all(D_eigenvalues > 0))

print('Mean Diffusivity:')
print(D_eigenvalues.mean())

# ============================================================
# 4. Inspect the MRI measurement settings
# ============================================================

print("\n--- MEASUREMENT SETTINGS ---")
print("Shape of b-values:", gtab.bvals.shape)
print("Shape of b-vectors:", gtab.bvecs.shape)

print("\nFirst 20 b-values:")
print(gtab.bvals[:20])

print("\nFirst five b-vectors:")
print(gtab.bvecs[:5])


# ============================================================
# 5. Construct q
#
# q corresponds to x in the equation:
#
# S(x) = S0 * exp(-x^T D x)
# ============================================================

q = np.sqrt(gtab.bvals[:, None]) * gtab.bvecs

print("\n--- MODEL INPUT q ---")
print("Shape of q:", q.shape)
print("First five rows of q:")
print(q[:5])


# ============================================================
# 6. Predict all 160 MRI signals
# ============================================================

quadratic_terms = np.einsum(
    "ni,ij,nj->n",
    q,
    D_init,
    q,
)

predicted_y = S0_init * np.exp(-quadratic_terms)

print("\n--- PREDICTED SIGNALS ---")
print("Shape of predicted_y:", predicted_y.shape)

print("\nFirst ten measured signals:")
print(y[:10])

print("\nFirst ten predicted signals:")
print(predicted_y[:10])


# ============================================================
# 7. Inspect measurements with nonzero diffusion weighting
# ============================================================

nonzero_indices = np.where(gtab.bvals > 0)[0]
selected_indices = nonzero_indices[:10]

print("\n--- FIRST TEN DIFFUSION-WEIGHTED MEASUREMENTS ---")
print("Measurement indices:")
print(selected_indices)

print("\nb-values:")
print(gtab.bvals[selected_indices])

print("\nMeasured signals:")
print(y[selected_indices])

print("\nPredicted signals:")
print(predicted_y[selected_indices])


# ============================================================
# 8. Calculate simple prediction errors
# ============================================================

errors = y - predicted_y

mean_error = np.mean(errors)
rmse = np.sqrt(np.mean(errors**2))

print("\n--- PREDICTION ERRORS ---")
print("Mean error:", mean_error)
print("Root mean squared error:", rmse)
print("Assumed noise standard deviation:", 29)


# ============================================================
# 9. Plot measured and predicted signals
# ============================================================

plt.figure(figsize=(11, 5))

plt.plot(
    y,
    "o",
    markersize=4,
    alpha=0.7,
    label="Measured signal",
)

plt.plot(
    predicted_y,
    "-",
    linewidth=2,
    label="Predicted signal",
)

plt.xlabel("Measurement number")
plt.ylabel("MRI signal")
plt.title("Measured and predicted MRI signals")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()