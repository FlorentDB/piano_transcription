import numpy as np

# Load the .npy file
data = np.load('checkpoints/losses.npy', allow_pickle=True)

# Print the data
print(data)
