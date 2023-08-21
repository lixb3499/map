import numpy as np
from scipy.optimize import linear_sum_assignment

# Example distance matrix (cost matrix)
distance_matrix = np.array([
    [1, 8, 3],
    [3, 1, 8],
    [9, 3, 1],
    [8, 3, 0.5]
])

# Convert the cost matrix to its negative form to transform the problem into a maximization problem
maximize_matrix = distance_matrix

# Solve the assignment problem
row_indices, col_indices = linear_sum_assignment(maximize_matrix)

# Print the indices of the matched pairs
for worker, job in zip(row_indices, col_indices):
    print(f"Worker {worker} is assigned to Job {job}")

# Calculate the total cost (negative value of the maximum sum)
total_cost = -maximize_matrix[row_indices, col_indices].sum()
print("Total cost:", total_cost)

# import numpy as np
# from scipy.optimize import linear_sum_assignment
#
# # Example distance matrix (cost matrix) with different numbers of workers and jobs
# distance_matrix = np.array([
#     [8, 5, 9, 6],
#     [4, 7, 3, 8],
#     [2, 6, 5, 7]
# ])
#
# # Set a distance threshold
# distance_threshold = 6
#
# # Apply the distance threshold to the distance matrix
# distance_matrix[distance_matrix <= distance_threshold] = distance_matrix.max() + 1
#
# # Convert the cost matrix to its negative form to transform the problem into a maximization problem
# maximize_matrix = distance_matrix
#
# # Solve the assignment problem
# row_indices, col_indices = linear_sum_assignment(maximize_matrix)
#
# # Print the indices of the matched pairs
# for worker, job in zip(row_indices, col_indices):
#     if distance_matrix[worker][job] <= distance_threshold:
#         print(f"Worker {worker} is assigned to Job {job} with distance {distance_matrix[worker][job]}")
#
# # Calculate the total cost (negative value of the maximum sum)
# total_cost = maximize_matrix[row_indices, col_indices].sum()
# print("Total cost:", total_cost)
