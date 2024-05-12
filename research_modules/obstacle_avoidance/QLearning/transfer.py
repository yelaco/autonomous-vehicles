import numpy as np
import pickle

# Define the dimensions of the old and new Q-tables
old_dims = [3, 3, 4, 4, 3]
new_dims = [3, 3, 4, 4, 2, 3]

# Load the old Q-table from a pickle file
with open('q_table.pkl', 'rb') as file:
    old_qtable = pickle.load(file)

with open('new_q_table.pkl', 'rb') as file:
    new_qtable = pickle.load(file)
    
# Copy elements from old Q-table to new Q-table using a loop
for i in range(old_dims[0]):
    for j in range(old_dims[1]):
        for k in range(old_dims[2]):
            for l in range(old_dims[3]):
                for m in range(old_dims[4]):
                    new_qtable[i, j, k, l, 0, m] = old_qtable[i, j, k, l, 0, m]

# Save the new Q-table to a new pickle file
with open('new_new_q_table.pkl', 'wb') as file:
    pickle.dump(new_qtable, file)

# Display the old and new Q-table shapes
print("Old Q-table shape:", old_qtable.shape)
print("New Q-table shape:", new_qtable.shape)
