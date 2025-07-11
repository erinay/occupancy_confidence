import numpy as np
from scipy import ndimage


def Filtered_Occupancy(occupancy_data, N_frames):
    """
    Filters the last N_frames of binary occupancy maps to reduce flickering.
    
    Parameters:
        occupancy_data (list of np.ndarray): List of 2D binary arrays (0 or 1).
        N_frames (int): Number of frames to use for filtering.

    Returns:
        normalized_confidence (np.ndarray): Normalized confidence values.
        filtered_occupancy (np.ndarray): Binary map where only persistent occupancy is kept.
    """
    occupancy_stack = np.stack(occupancy_data[-N_frames:], axis=0)  # shape (N, H, W)
    Confidence_values = np.sum(occupancy_stack, axis=0)  # Confidence_values are the sum of occupancy over the last N frames.
    normalized_confidence = Confidence_values / N_frames  # Normalize confidence values
    filtered_occupancy = (normalized_confidence  == 1).astype(int) # Keep only the pixels that are occupied in all N frames.
    return normalized_confidence, filtered_occupancy



def Filtered_Occupancy_Convolution(occupancy_data,  C_old, N_frames, num_rings=1, decay=0.2,):
    """
    Applies convolution-based filtering to reduce flickering in binary occupancy maps.

    Parameters:
        occupancy_data (list of np.ndarray): List of 2D binary arrays (0 or 1).
        N_frames (int): Number of frames to use for filtering.
        num_rings (int): Number of decay rings in the kernel.
        decay (float): Decay factor for each ring (e.g., 0.2 means 1.0, 0.8, 0.6...).

    Returns:
        buffered_binary_conv, normalized_confidence, filtered_occupancy
    """
    Confidence_values = np.copy(C_old) 
    occupancy_map = occupancy_data[-1]
    kernel = create_square_decay_kernel(kernel_size=5, decay=0.2)

    # Convolve the occupancy map with the kernel
    buffered_binary_conv = ndimage.convolve(occupancy_map, kernel, mode='constant', cval=0.0)

    Th = 1.0  # Convolution Threshold for strong responses
   
    # Increase confidence in regions with  above_convolution_threshold
    beta1 = 0.1
    sig1 = 1 - np.exp(-beta1 * buffered_binary_conv) # sigmoid
    mask1 = buffered_binary_conv > Th

    C_plus = 1.0  # Confidence boost for strong responses
    Confidence_values[mask1] = (1 - sig1[mask1]) * C_old[mask1] + sig1[mask1] * C_plus  # Boost strong responses

    # Decrease confidence in regions with below_convolution_threshold
    beta2 = 0.5
    k = 0.5
    sig2= 1 - np.exp(-beta2 * k / 2)  # sigmoid
    mask2 = buffered_binary_conv <= Th
    
    C_minus = 0.0 # Confidence reduction for weak responses`
    # Confidence_values[mask2] = (1 - sig[mask2]) * C_old[mask2] + sig[mask2] * C_minus  
    Confidence_values[mask2] = (1 - sig2) * C_old[mask2] + sig2 * C_minus  
    
    # Normalize and threshold
    filtered_occupancy = (Confidence_values >= 0.7).astype(int)
    return buffered_binary_conv, Confidence_values, filtered_occupancy


def create_square_decay_kernel(kernel_size=3, decay=0.2):
    """
    Create a square convolution kernel where each concentric Chebyshev ring decays linearly.

    Parameters:
        kernel_size (int): Size of the square kernel (must be odd, e.g., 3, 5, 7).
        decay (float): Linear decay applied per ring (e.g., 0.2 gives values 1.0, 0.8, 0.6...).

    Returns:
        kernel (np.ndarray): 2D convolution kernel.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer.")
    
    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    for i in range(kernel_size):
        for j in range(kernel_size):
            ring = max(abs(i - center), abs(j - center))  # Chebyshev distance
            value = max(0.0, 1.0 - decay * ring)
            kernel[i, j] = value

    return kernel


# kernel = create_square_decay_kernel(kernel_size=3, decay=0.5)
# print(kernel)