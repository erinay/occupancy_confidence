import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
from utils import Filtered_Occupancy, Filtered_Occupancy_Convolution


# Load the CSV file containing occupancy data
df = pd.read_csv("_intensity_map.csv")

# Set these to your data dimensions
HEIGHT = 120  # example height
WIDTH = 120   # example width
N_frames = 5  # Number of frames for filtering


plt.ion()
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
axs = axs.flatten()  # Convert 2x3 array of Axes into flat list

# Initialize the images with zeros
im0 = axs[0].imshow(np.zeros((HEIGHT, WIDTH)), cmap='gray', vmin=0, vmax=1, interpolation='none', origin='lower')
im1 = axs[1].imshow(np.zeros((HEIGHT, WIDTH)), cmap='hot', interpolation='none', origin='lower', vmin=0, vmax=1)
im2 = axs[2].imshow(np.zeros((HEIGHT, WIDTH)), cmap='gray', vmin=0, vmax=1, interpolation='none', origin='lower')
# Initialize the images with zeros
im3 = axs[3].imshow(np.zeros((HEIGHT, WIDTH)), cmap='viridis', interpolation='none', origin='lower', vmin=0, vmax=3,)
im4 = axs[4].imshow(np.zeros((HEIGHT, WIDTH)), cmap='hot', interpolation='none', origin='lower', vmin=0, vmax=1,)
im5 = axs[5].imshow(np.zeros((HEIGHT, WIDTH)), cmap='gray', vmin=0, vmax=1, interpolation='none', origin='lower')

# Colorbar for confidence values
cbar = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
cbar = fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)
cbar = fig.colorbar(im4, ax=axs[4], fraction=0.046, pad=0.04)

axs[0].set_title(f"Buffered Binary Frame\nFrame 0")

axs[1].set_title(f"Confidence Values\n(Sum over last {N_frames} frames)")
axs[2].set_title("Filtered Occupancy Map\nFiltering: INACTIVE")

axs[3].set_title(f"Convolution on Buffered Binary Frame\nFrame 0")
axs[4].set_title("Confidence Values \n from Convolution on (Current Frame)")
axs[5].set_title(f"Filtered Occupancy Map\nFiltering: ACTIVE")

Confidence_values_conv = 0
while True:
# for i in range(1):
    # Initialize buffer and plotting
    BUFFERED_BINARY_FRAMES = []

    for i, msg in enumerate(df["data"]):

        C_old_conv = Confidence_values_conv
        # Parse data string from msg
        index = msg.find("data=")
        array_start = index + 6
        occupancy_str = msg[array_start:-2]
        values = [int(x) for x in occupancy_str.split(',')]

        occupancy_arr = np.array(values).reshape(HEIGHT, WIDTH)
        buffered_binary = ndimage.binary_dilation(occupancy_arr, iterations=1).astype(occupancy_arr.dtype)
        # buffered_binary = (occupancy_arr > 0).astype(int)  # Convert to binary occupancy map
        # Append the current binary frame to the buffer
        BUFFERED_BINARY_FRAMES.append(buffered_binary)

        # If buffer exceeds N_frames, pop the oldest frame and filter
        # Use the last N frames to smooth/stabilize
        if len(BUFFERED_BINARY_FRAMES) > N_frames:
            BUFFERED_BINARY_FRAMES.pop(0)
            Confidence_values, filtered_binary = Filtered_Occupancy(BUFFERED_BINARY_FRAMES, N_frames)
            buffered_binary_conv, Confidence_values_conv, filtered_binary_conv = Filtered_Occupancy_Convolution(BUFFERED_BINARY_FRAMES, C_old_conv, N_frames)
            enough_frames = 1
        else:
            # If not enough frames, use the current buffered binary as both confidence and filtered
            # This will show the current state without filtering
            Confidence_values, filtered_binary = buffered_binary, buffered_binary
            buffered_binary_conv, Confidence_values_conv, filtered_binary_conv = Filtered_Occupancy_Convolution(BUFFERED_BINARY_FRAMES, C_old_conv, N_frames)
            enough_frames = 0

        # Update images data and titles
        im0.set_data(buffered_binary)
        axs[0].set_title(f"Buffered Binary Frame\nFrame {i}")
        
        # 1. Confidence and filtered occupancy maps: Total number of frames
        im1.set_data(Confidence_values)
        axs[1].set_title(f"Confidence Values\n(Sum over last {N_frames} frames)")
        im2.set_data(filtered_binary)
        status = "ACTIVE" if enough_frames else "INACTIVE"
        axs[2].set_title(f"Filtered Occupancy Map\nFiltering: {status}")

        # 2. Confidence and filtered occupancy maps: Total number of frames
        im3.set_data(buffered_binary_conv)
        axs[3].set_title(f"Convolution on Buffered Binary Frame\nFrame {i}")
        im4.set_data(Confidence_values_conv)
        axs[4].set_title("Confidence Values \n from Convolution on (Current Frame)")
        im5.set_data(filtered_binary_conv)
        status = "ACTIVE"
        axs[5].set_title(f"Filtered Occupancy Map\nFiltering: {status}")
        
        
        # Turn off axis ticks for all plots
        for ax in axs:
            ax.axis("off")

        plt.pause(0.05)

# plt.ioff()
# plt.show()
