import numpy as np
mask = np.array([0, 0, 0, 0, 1], dtype=np.float32)  # Masque binaire
invalid = np.array([1, 0, 0, 0, 0], dtype=np.float32)
print(mask - invalid)