import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates

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



def Filtered_Occupancy_Convolution(occupancy_data,  C_old, N_frames, q_rel, num_rings=1, decay=0.2):
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
    occupancy_map = occupancy_data[-1]
    kernel = create_square_decay_kernel(kernel_size=5, decay=0.2)
    # TF old points to current frame
    if q_rel!=None:
        C_old_tf = rotate_occupancy_map(C_old, q_rel)
        Confidence_values = np.copy(C_old_tf)
    else: 
        C_old_tf = C_old
        Confidence_values = np.copy(C_old)

    # Convolve the occupancy map with the kernel
    buffered_binary_conv = ndimage.convolve(occupancy_map, kernel, mode='constant', cval=0.0)
    Th = 1.0  # Convolution Threshold for strong responses
   
    # Increase confidence in regions with  above_convolution_threshold
    beta1 = 0.1
    sig1 = 1 - np.exp(-beta1 * buffered_binary_conv) # sigmoid
    mask1 = buffered_binary_conv > Th

    C_plus = 1.0  # Confidence boost for strong responses
    Confidence_values[mask1] = (1 - sig1[mask1]) * C_old_tf[mask1] + sig1[mask1] * C_plus  # Boost strong responses

    # Decrease confidence in regions with below_convolution_threshold
    beta2 = 0.5
    k = 0.5
    sig2= 1 - np.exp(-beta2 * k / 2)  # sigmoid
    mask2 = buffered_binary_conv <= Th
    
    C_minus = 0.0 # Confidence reduction for weak responses`
    # Confidence_values[mask2] = (1 - sig[mask2]) * C_old[mask2] + sig[mask2] * C_minus  
    Confidence_values[mask2] = (1 - sig2) * C_old_tf[mask2] + sig2 * C_minus  
    
    # Normalize and threshold
    filtered_occupancy = (Confidence_values >= 0.01).astype(int)
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

def decay_masked(pc_data, occupancy_map, visibility_mask, q_rel, decay_rate=0.5,decay_rate_background=0.95):
    if q_rel!=None:
        occupancy_map_tf = rotate_occupancy_map(occupancy_map, q_rel, verbose=True)
    else: 
        occupancy_map_tf = occupancy_map

    # Grow visibility mask area
    kernel = np.ones((30, 30), np.uint8)  # 5x5 square kernel
    grown_mask = ndimage.convolve(visibility_mask.astype(np.uint8), kernel,  mode='constant', cval=0.0).astype(bool)
    background_mask = ~grown_mask

    if np.count_nonzero(grown_mask)>0:
        occupancy_map_tf[grown_mask] *= decay_rate
        occupancy_map_tf[grown_mask] += pc_data.astype(float)[grown_mask]
    
    occupancy_map[background_mask]*=decay_rate_background
    # print(np.amax(occupancy_map))

    return occupancy_map_tf

def decay_occupancy(pc_data, occupancy_map, q_rel, decay_rate=0.5, alpha=1.0):
    if q_rel!=None:
        occupancy_map_tf = rotate_occupancy_map(occupancy_map, q_rel, verbose=True)
    else: 
        occupancy_map_tf = occupancy_map
    occupancy_map_tf *= decay_rate
    occupancy_map_tf += pc_data.astype(float)

    return occupancy_map_tf

def interpolate_pose(imu_poses, target_time):
    """
    Linearly interpolate pose at a given timestamp using IMU data.

    Parameters:
        imu_poses (list of dict): Each entry must have:
            {
                'timestamp': float,
                'position': np.array([x, y]),
                'theta': float  # orientation in radians
            }
        target_time (float): Desired timestamp for interpolation.

    Returns:
        dict: {'position': np.array([x, y]), 'theta': float}
    """
    if not imu_poses or target_time < imu_poses[0]['timestamp'] or target_time > imu_poses[-1]['timestamp']:
        raise ValueError("Target time outside IMU timestamp range.")

    # Find the two surrounding poses
    for i in range(len(imu_poses) - 1):
        t0 = imu_poses[i]['timestamp']
        t1 = imu_poses[i + 1]['timestamp']

        if t0 <= target_time <= t1:
            pose0 = imu_poses[i]
            pose1 = imu_poses[i + 1]

            alpha = (target_time - t0) / (t1 - t0)

            # Linear interpolate position
            pos_interp = (1 - alpha) * pose0['position'] + alpha * pose1['position']

            # Interpolate theta taking care of angle wrapping
            theta0 = pose0['theta']
            theta1 = pose1['theta']
            dtheta = np.arctan2(np.sin(theta1 - theta0), np.cos(theta1 - theta0))
            theta_interp = theta0 + alpha * dtheta

            return {'position': pos_interp, 'theta': theta_interp}

    raise RuntimeError("Interpolation failed; timestamp not bracketed.")

def find_closest(imu_timestamp, t_pc):
    """
    Grabs index of closest timestamp to t_pc in imu_df
    """
    valid = imu_timestamp[imu_timestamp <= t_pc]
    if valid.empty:
        return None  # no timestamp before t_pc found
    # Return the index of the max timestamp ≤ t_pc
    closest_index = valid.idxmax()
    return closest_index

def rotate_occupancy_map(occupancy_map: np.ndarray, Q_rel: R, verbose=False) -> np.ndarray:
    """
    Rotates a 2D confidence map using yaw from relative quaternion.

    Parameters:
        occupancy_map (np.ndarray): Float 2D array (HxW), confidence values [0, 1].
        Q_rel (scipy.spatial.transform.Rotation): Relative rotation between frames.

    Returns:
        np.ndarray: Rotated confidence map of same shape.
    """
    # Extract yaw angle from quaternion
    yaw = Q_rel.as_euler('zyx', degrees=True)[2]  # degrees=True for scipy rotate
    euler = Q_rel.as_euler('zyx', degrees=True)
    if verbose:
        print(f"Yaw (Z): {euler[0]:.2f}, Pitch (Y): {euler[1]:.2f}, Roll (X): {euler[2]:.2f}")

    # Rotate the image around its center without changing shape
    rotated = ndimage.rotate(
        occupancy_map,
        angle=-yaw,  # Negative for correct direction
        reshape=False,
        order=1,         # Bilinear interpolation
        mode='constant',
        cval=0.0         # Fill empty space with 0 (no confidence)
    )

    return rotated


def evolve(occupancy_map, flow, dt=0.02):
    '''
    Estimates next confidence map at next time-step
    '''
    H,W = occupancy_map.shape
    fx = flow[:, :, 0]
    fy = flow[:, :, 1]

    # Create a grid of pixel indices
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # New (x, y) positions after dt
    x_new = x + fx * dt
    y_new = y + fy * dt

    # Sample map_t at new positions (backward warping)
    coords = np.stack([y_new.ravel(), x_new.ravel()])
    propagated_map = map_coordinates(occupancy_map, coords, order=1, mode='constant', cval=0.0)
    return propagated_map.reshape(H, W)
