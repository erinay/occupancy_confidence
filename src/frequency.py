import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
import cv2

def fft(occupancy_data, N_frames):
    occupancy_stack = np.stack(occupancy_data[-N_frames:], axis=0)  # shape (N, H, W)
    return np.fft.fft(occupancy_stack, axis=0)

# Convert desired frequency range to FFT bin indices
def frequency_to_bin_indices(f_min, f_max, sampling_rate, N_frames):
    freqs = np.fft.fftfreq(N_frames, d=1/sampling_rate)  # e.g., [0, 1.25, ..., -1.25]
    freqs = np.abs(freqs)  # ignore negative freqs
    return np.where((freqs >= f_min) & (freqs <= f_max))[0]

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
    filtered_occupancy = (Confidence_values >= N_frames*0.4*100).astype(int) # Keep only the pixels that are occupied in all N frames.
    normalized_confidence = Confidence_values / N_frames  # Normalize confidence values
    return normalized_confidence, filtered_occupancy

# Load CSV
df = pd.read_csv("data/accum_hits/_intensity_map.csv")

HEIGHT = 120
WIDTH = 120
N_frames = 5
BUFFERED_BINARY_FRAMES = []
sampling_rate = 10.0
prev_confidence = None

# Set your target frequency range
target_freq_range = (1.75, 4.0)
target_bins = frequency_to_bin_indices(*target_freq_range, sampling_rate, N_frames)
print("Target frequency bins:", target_bins)

# Set up live plotting
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
dominant_freq_im = ax.imshow(np.zeros((HEIGHT, WIDTH)), cmap='plasma', vmin=0, vmax=N_frames-1, origin='lower')
cbar = fig.colorbar(dominant_freq_im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Dominant Frequency Index")

ax.set_title("Dominant Flicker Frequency per Pixel")
ax.axis("off")


fig2, ax2 = plt.subplots(figsize=(5, 5))
im0 = ax2.imshow(np.zeros((HEIGHT, WIDTH)), cmap='gray', vmin=0, vmax=1, interpolation='none', origin='lower')
ax.axis("off")

fig3, ax3 = plt.subplots(figsize=(5,5))
im1 = ax3.imshow(np.zeros((HEIGHT, WIDTH)), cmap='gray', vmin=0, vmax=1, interpolation='none', origin='lower')
ax.axis("off")

fig4, ax4 = plt.subplots(figsize=(6, 6))
im2 = ax4.imshow(np.zeros((HEIGHT, WIDTH)), cmap='gray', vmin=0, vmax=1, interpolation='none', origin='lower')

mask = np.zeros((120,120))
for i, msg in enumerate(df["data"]):
    # Parse occupancy map
    index = msg.find("data=")
    array_start = index + 6
    occupancy_str = msg[array_start:-2]
    values = [int(x) for x in occupancy_str.split(',')]
    occupancy_arr = np.array(values).reshape(HEIGHT, WIDTH)

    # Optional: dilate
    # buffered_binary = ndimage.binary_dilation(occupancy_arr, iterations=1).astype(occupancy_arr.dtype)
    
    BUFFERED_BINARY_FRAMES.append(occupancy_arr)
    if len(BUFFERED_BINARY_FRAMES) > N_frames:
        BUFFERED_BINARY_FRAMES.pop(0)
        ### Trial 3: update incoming data with previous fft
        latest = BUFFERED_BINARY_FRAMES[-1].copy()
        flicker_pixels = np.zeros_like(latest)
        flicker_pixels[mask] = latest[mask] 
        dilated = ndimage.binary_dilation(flicker_pixels, iterations=1).astype(np.uint8)
        # Merge dilated result back into the latest frame
        latest[dilated == 1] = 1
        # Update the buffered frame in-place
        BUFFERED_BINARY_FRAMES[-1] = latest


    if len(BUFFERED_BINARY_FRAMES) == N_frames:
        fft_result = fft(BUFFERED_BINARY_FRAMES, N_frames)
        mag = np.abs(fft_result)  # shape (N_frames, H, W)

        # Get argmax frequency index per pixel
        dominant_freq_idx = np.argmax(mag, axis=0)  # shape (H, W)

        # Update plot
        dominant_freq_im.set_data(dominant_freq_idx)
        ax.set_title(f"Dominant Frequency per Pixel: Frame {i}")
        plt.pause(0.05)

        # Create a mask for pixels whose dominant frequency falls in target range
        mask = np.isin(dominant_freq_idx, target_bins)
        # print(mask.shape)
        # ### TRIAL 1: Update last 
        # latest = BUFFERED_BINARY_FRAMES[-1].copy()
        # flicker_pixels = np.zeros_like(latest)
        # flicker_pixels[mask] = latest[mask]  # keep only target pixels

        # dilated = ndimage.binary_dilation(flicker_pixels, iterations=2).astype(np.uint8)

        # # Merge dilated result back into the latest frame
        # latest[dilated == 1] = 1

        # # Update the buffered frame in-place
        # BUFFERED_BINARY_FRAMES[-1] = latest

        # ### TRIAL 2: UPDATE all masks
        # Step 4: Loop through ALL buffered frames and apply dilation only on masked pixels
        # for k in range(N_frames):
        #     original = BUFFERED_BINARY_FRAMES[k]
        #     flicker_pixels = np.zeros_like(original)
        #     flicker_pixels[mask] = original[mask]

        #     # Apply dilation only at those regions
        #     dilated = ndimage.binary_dilation(flicker_pixels, iterations=1).astype(np.uint8)

        #     # Merge back into frame
        #     modified = original.copy()
        #     modified[dilated == 1] = 1

        #     BUFFERED_BINARY_FRAMES[k] = modified  # update in-place


        Confidence_values, filtered_binary = Filtered_Occupancy(BUFFERED_BINARY_FRAMES, N_frames)
        if prev_confidence is not None:
            # Convert to float32
            prev_conf = prev_confidence.astype(np.float32)
            curr_conf = Confidence_values.astype(np.float32)

            # Compute Farneback optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev=prev_conf,
                next=curr_conf,
                flow=None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            # Extract flow vectors
            flow_x, flow_y = flow[..., 0], flow[..., 1]
            
            # Visualize the flow as a quiver plot
            step = 5  # downsample for visualization
            y, x = np.mgrid[0:HEIGHT:step, 0:WIDTH:step]
            u = flow_x[::step, ::step]
            v = flow_y[::step, ::step]

            ax4.clear()  # <-- This clears the previous plot
            ax4.imshow(curr_conf, cmap='gray', origin='lower')
            ax4.quiver(x, y, u, v, color='red', angles='xy', scale_units='xy', scale=1)
            ax4.set_title(f"Optical Flow on Confidence Map – Frame {i}")
            ax4.axis("off")
            plt.pause(0.1)
        # Update images data

        im0.set_data(Confidence_values)
        # Update titles
        ax2.set_title(f"Confidence Values\n(Sum over last {N_frames} frames)")

        im1.set_data(filtered_binary)
        # Update titles
        ax3.set_title(f"Filtered Binay\n(Over last {N_frames} frames)")
        prev_confidence = Confidence_values.copy()


plt.ioff()
plt.show()
