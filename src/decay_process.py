import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from utils import *

def visualize_decay_masked(pc_data, occupancy_map, visibility_mask, q_rel=None,
                           decay_rate=0.5, decay_rate_background=0.95):
    # Step 1: Rotate occupancy map (if needed)
    if q_rel is not None:
        occupancy_map_tf = rotate_occupancy_map(occupancy_map, q_rel)
    else:
        occupancy_map_tf = occupancy_map.copy()

    # Step 2: Grow visibility mask
    kernel = np.ones((30, 30), np.uint8)
    grown_mask = ndimage.convolve(visibility_mask.astype(np.uint8), kernel, mode='constant', cval=0.0).astype(bool)
    background_mask = ~grown_mask

    # Step 3: Apply decay and updates
    updated_map = occupancy_map_tf.copy()
    if np.count_nonzero(grown_mask) > 0:
        updated_map[grown_mask] *= decay_rate
        updated_map[grown_mask] += pc_data.astype(float)[grown_mask]

    updated_map[background_mask] *= decay_rate_background

    # Step 4: Plot everything
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    axs[0, 0].imshow(pc_data, cmap='viridis',origin='lower')
    axs[0, 0].set_title("pc_data (current visibility)")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(visibility_mask, cmap='gray',origin='lower')
    axs[0, 1].set_title("Raw visibility_mask (t=1)")
    axs[0, 1].axis('off')

    axs[0, 2].imshow(grown_mask, cmap='gray',origin='lower')
    axs[0, 2].set_title("Grown mask (convolved)")
    axs[0, 2].axis('off')

    axs[1, 0].imshow(background_mask, cmap='gray',origin='lower')
    axs[1, 0].set_title("Background mask")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(occupancy_map, cmap='magma',origin='lower')
    axs[1, 1].set_title("occupancy_map (before decay)")
    axs[1, 1].axis('off')

    axs[1, 2].imshow(updated_map, cmap='magma',origin='lower')
    axs[1, 2].set_title("Updated occupancy map (after decay)")
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()
