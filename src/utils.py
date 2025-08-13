import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates, label, center_of_mass, shift
import cv2

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



def Filtered_Occupancy_Convolution(occupancy_data,  C_old, num_rings=1, decay=0.2):
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
    occupancy_map = occupancy_data
    kernel = create_square_decay_kernel(kernel_size=5, decay=0.2)
    # TF old points to current frame 
    Confidence_values = np.copy(C_old)

    # Convolve the occupancy map with the kernel
    buffered_binary_conv = ndimage.convolve(occupancy_map, kernel, mode='constant', cval=0.0)
    Th = 1.0  # Convolution Threshold for strong responses
   
    # Increase confidence in regions with  above_convolution_threshold
    beta1 = 0.1
    sig1 = 1 - np.exp(-beta1 * buffered_binary_conv) # sigmoid
    mask1 = buffered_binary_conv > Th

    C_plus = 2.0  # Confidence boost for strong responses
    Confidence_values[mask1] = (1 - sig1[mask1]) * C_old[mask1] + sig1[mask1] * C_plus  # Boost strong responses

    # Decrease confidence in regions with below_convolution_threshold
    beta2 = 0.5
    k = 0.5
    sig2= 1 - np.exp(-beta2 * k / 2)  # sigmoid
    mask2 = buffered_binary_conv <= Th
    
    C_minus = 0# Confidence reduction for weak responses`
    # Confidence_values[mask2] = (1 - sig[mask2]) * C_old[mask2] + sig[mask2] * C_minus  
    Confidence_values[mask2] = (1 - sig2) * C_old[mask2] + sig2 * C_minus  
    
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

def Filtered_Occupancy_Convolution_Masked(occupancy_data, C_old):
    Confidence_values = np.copy(C_old)
    kernel = create_square_decay_kernel(kernel_size=5, decay=0.2)
    
    # Buffered binary convolution
    buffered_binary_conv = ndimage.convolve(occupancy_data, kernel, mode='constant', cval=0.0)
    
    # Grow visibility mask area
    visibility_mask = (buffered_binary_conv>0)
    kernel = np.ones((30, 30), np.uint8)  # 5x5 square kernel
    grown_mask = ndimage.convolve(visibility_mask.astype(np.uint8), kernel,  mode='constant', cval=0.0).astype(bool)
    background_mask = ~grown_mask

    # Increase confidence in regions with  above_convolution_threshold
    Th = 1.0  # Convolution Threshold for strong responses
    # Increase confidence in regions with  above_convolution_threshold
    beta1 = 0.1
    sig1 = 1 - np.exp(-beta1 * buffered_binary_conv) # sigmoid
    mask1 = buffered_binary_conv > Th
    C_plus = 2.0  # Confidence boost for strong responses
    Confidence_values[mask1] = (1 - sig1[mask1]) * C_old[mask1] + sig1[mask1] * C_plus  # Boost strong responses

    # Decrease confidence in regions with below_convolution_threshold & inside visbility mask
    beta2 = 0.5
    k = 0.5
    sig2= 1 - np.exp(-beta2 * k / 2)  # sigmoid
    mask2 = (buffered_binary_conv <= Th) & grown_mask
    
    C_minus = -1 # Confidence reduction for weak responses`
    Confidence_values[mask2] = (1 - sig2) * C_old[mask2] + sig2 * C_minus  
    
    # Decrase confidence in regions in background by lesser amount
    C_minus = 0.0 # Confidence reduction for weak responses`
    sig3 = 0.05
    Confidence_values[background_mask] = (1 - sig3) * C_old[background_mask]   
    
    return buffered_binary_conv, Confidence_values    

def decay_occupancy(pc_data, occupancy_map, decay_rate=0.5, alpha=1.0):
    occupancy_map *= decay_rate
    occupancy_map += pc_data.astype(float)

    return occupancy_map

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

def evolve(confidence_map, labels, blobs, dt=0.02):
    '''
    Estimates next confidence map at next time-step
    '''
    H, W = confidence_map.shape
    predicted_map = np.zeros_like(confidence_map)

    for lab, blob in enumerate(blobs, start=1):
        cluster_mask = (labels == lab)
        if cluster_mask.sum() == 0:
            continue

        avg_vx = blob['avg_vx']
        avg_vy = blob['avg_vy']

        # Shift the entire blob mask by velocity * dt
        shift_x = avg_vx
        shift_y = avg_vy

        # Extract the blob's confidence submap
        blob_conf = confidence_map * cluster_mask

        # Shift using interpolation = 0 (nearest) so we don't smear
        shifted_blob = shift(blob_conf, shift=(shift_y, shift_x), order=0, mode='constant', cval=0.0)

        # Add the shifted blob to the predicted map
        predicted_map = np.maximum(predicted_map, shifted_blob)

    predicted_map = np.clip(predicted_map, 0, 1)
    return predicted_map

def get_motion_blobs(vx_s, vy_s, confidence_map, conf_threshold=0.05, speed_threshold=0.0001):
    speed = np.hypot(vx_s, vy_s)
    H,W = confidence_map.shape
    labels = np.zeros((H,W))

    # Build mask of valid cells
    # mask = (confidence_map >= conf_threshold) #& (speed >= 0.0001)
    mask = (speed>1e-5)

    # If mask is empty, return empty list immediately
    if np.sum(mask) == 0:
        return labels,[]

    # Connected components labeling (8-connectivity)
    structure = np.ones((3,3), dtype=int)
    labels, num_labels = label(mask, structure=structure)
    if num_labels == 0:
        return labels,[]
    blobs = []
    for lab in range(1, num_labels + 1):
        region_mask = (labels == lab)
        conf_region = confidence_map[region_mask]
        weight_sum = np.sum(conf_region)

        if weight_sum == 0:
            continue  # skip zero-confidence blobs

        vx_region = vx_s[region_mask]
        vy_region = vy_s[region_mask]

        avg_vx = np.sum(vx_region * conf_region) / weight_sum
        avg_vy = np.sum(vy_region * conf_region) / weight_sum

        centroid = center_of_mass(region_mask)

        blobs.append({
            'centroid': centroid,
            'avg_vx': avg_vx,
            'avg_vy': avg_vy,
            'size': np.sum(region_mask),
            'label_mask': region_mask
        })

    return labels, blobs

def evolve_map(conf_map, u, v):
    H, W = conf_map.shape
    # Create grid of coordinates in new frame
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

    # Calculate source coordinates in old frame for backward warp
    src_x = grid_x + u +0.5
    src_y = grid_y + v+0.5

    # Clip coordinates to valid range
    src_x = np.clip(src_x, 0, W-1)
    src_y = np.clip(src_y, 0, H-1)

    # Use cv2.remap for interpolation (expects float32)
    old_conf_map_f = conf_map.astype(np.float32)
    conf_current_frame = cv2.remap(old_conf_map_f, src_x.astype(np.float32), src_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
 
    return conf_current_frame

def filter(predicted_map, confidence_map, alpha=0.7):
    updated_map = alpha*predicted_map + (1-alpha)*confidence_map
    return updated_map

def thresholding_std(conf_map, predicted_map, sigma_factor=1.0):
    # Statistical Information
    mu = np.mean(conf_map)
    sigma = np.std(conf_map)

    # Thresholding
    T_hi = mu + sigma_factor * sigma
    T_lo = 2*mu - sigma_factor * sigma

    # Clip thresholds into valid range
    T_hi = np.clip(T_hi, 0, 1)
    T_lo = np.clip(T_lo, 0, 1)

    # --- Step 2: Apply initial binary masks ---
    strong = conf_map >= T_hi
    weak = (conf_map >= T_lo) & (conf_map < T_hi)

    kernel = np.ones((2,2))
    strong_nbhd = ndimage.convolve(strong, kernel,mode='constant',cval=0)
     # Promote weak pixels that are near any strong pixel
    grow_mask = (strong_nbhd > 0) & weak

    binary_map = strong | grow_mask
    return binary_map.astype(np.uint8), (T_lo, T_hi)
