# # intro to CNNs
## Layer 1: convolution layer
 # converts images into an array of numbers and does dimensionality reduction of array , that doesnt let the model lose important info as well as complexity is reduced


## Layer 2: Relu - regularization rectified unit
# converts all negatives into 0 and positives as is , because it doesnt make sense to consider negative values since they are already dark/dead pixels , so doesnt matter if it all a  negative or a zero since it is all dark , Range is [0,infinity]

## Layer 3: Pooling layer
# Reduce the dimensionality , helps control overfitting , filters over 2x2 matrix
