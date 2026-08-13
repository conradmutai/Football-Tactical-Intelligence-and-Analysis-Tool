import numpy as np

from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment


def wasserstein_distance(m_1, m_2, c_1, c_2):  # mean player 1/2, covariance player 1/2
    # ensuring they are np arrays
    m_1, m_2 = np.asarray(m_1), np.asarray(m_2)
    c_1, c_2 = np.asarray(c_1), np.asarray(c_2)

    # || m1 - m2 || ^ 2
    mean_diff_sq = np.sum(np.square(m_1 - m_2))

    # checking dimensions
    if c_2.ndim == 0 or c_2.size == 1:
        # Scalar (1D) simplification
        cov_term = c_1 + c_2 - 2 * np.sqrt(c_1 * c_2)
        w_dist_sq = mean_diff_sq + cov_term
    else:
        # calculates the square root for c_2
        sqrt_c_2 = sqrtm(c_2)

        # Handle small imaginary artifacts from numerical imprecision
        if np.iscomplexobj(sqrt_c_2):
            sqrt_c_2 = sqrt_c_2.real

        # handles the inner matrix calculations to feed into the general covariance calculations
        inner_matrix = sqrtm(sqrt_c_2 @ c_1 @ sqrt_c_2)
        if np.iscomplexobj(inner_matrix):
            inner_matrix = inner_matrix.real

        # calculates the covariance and applies trace to it
        covariance_calc = c_1 + c_2 - 2 * inner_matrix
        traced_covariance = np.trace(covariance_calc)

        w_dist_sq = mean_diff_sq + traced_covariance

    return np.sqrt(np.maximum(0.0, w_dist_sq))


def wasserstein_distance_squared(w_dist_matrix):
    # Computes total minimum squared Wasserstein distance for linear sum assignment.
    cost_matrix = np.square(np.asarray(w_dist_matrix))

    # scipy's implementation of the Hungarian / Munkres algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Sum of optimal pairings
    return cost_matrix[row_ind, col_ind].sum()


def bayesian_classification(mu_p_c, mu_p_o, sum_p_c, sum_p_o):  # mean and sum of player p and cluster/ mean and sum of player p and observation o
    mu_p_c, mu_p_o = np.asarray(mu_p_c), np.asarray(mu_p_o)
    sum_p_c, sum_p_o = np.asarray(sum_p_c), np.asarray(sum_p_o)

    # mean difference — this is the "x" the combined Gaussian is evaluated at
    diff = mu_p_c - mu_p_o

    # combined covariance
    combined_cov = sum_p_c + sum_p_o

    # dimensionality (2 for x,y pitch coordinates)
    k = diff.shape[0]

    # determinant and inverse of the combined covariance
    det_cov = np.linalg.det(combined_cov)
    inv_cov = np.linalg.inv(combined_cov)

    # normalizing constant: 1 / sqrt((2*pi)^k * |combined_cov|)
    norm_const = 1.0 / np.sqrt(((2 * np.pi) ** k) * det_cov)

    # exponent: -0.5 * diff^T * inv_cov * diff  (Mahalanobis distance squared)
    exponent = -0.5 * diff.T @ inv_cov @ diff

    return norm_const * np.exp(exponent)

