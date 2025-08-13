import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
from utils import *
import cv2
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R
import re
from decay_process import *
from scipy.ndimage import uniform_filter

# Load data
df = pd.read_csv("data/_map_convert.txt")

HEIGHT, WIDTH, N_frames = 120, 120, 5
step=1
scale_factor=5e2
# scale_factor=1e8*2.5
y_quiv, x_quiv = np.mgrid[0:HEIGHT:step, 0:WIDTH:step]

fig, axs = plt.subplots(2,3, figsize=(15,8))
axs = axs.flatten()

im0 = axs[0].imshow(np.zeros((HEIGHT,WIDTH)), cmap='gray', vmin=0, vmax=1,  origin='lower') # incoming data
im1 = axs[1].imshow(np.zeros((HEIGHT,WIDTH)), cmap='viridis', vmin=0, vmax=3,  origin='lower') # overlay velocity from "raw"
im2 = axs[2].imshow(np.zeros((HEIGHT,WIDTH)), cmap='viridis', vmin=0, vmax=3,  origin='lower') # predict next map with velocity data
im3 = axs[3].imshow(np.zeros((HEIGHT,WIDTH)), cmap='viridis', vmin=0, vmax=3,  origin='lower') # combine
im4 = axs[4].imshow(np.zeros((HEIGHT,WIDTH)), cmap='gray', vmin=0, vmax=1,  origin='lower') # convolution
im5 = axs[5].imshow(np.zeros((HEIGHT,WIDTH)), cmap='gray', vmin=0, vmax=1,  origin='lower')

# Define quiver
quiver = axs[3].quiver(x_quiv, y_quiv, np.zeros_like(x_quiv), np.zeros_like(y_quiv), 
                       color='red', scale=1, scale_units='xy', angles='xy')
for ax in axs:
    ax.axis("off")

# Global Variables
frame_init=2500
frame_index=frame_init
BUFFERED_BINARY_FRAMES=[]
prev_buffered_binary = np.zeros((HEIGHT,WIDTH))
prev_convolution = np.zeros((HEIGHT,WIDTH))

# necessary for current function definitions (i think?, i.e. not necessarily used)
Confidence_values_conv = np.zeros((HEIGHT, WIDTH))
C_old_conv = np.zeros((HEIGHT, WIDTH))  # Initialize confidence values for convolution
confidence_combined = np.zeros((HEIGHT,WIDTH))
binary_map = np.zeros((HEIGHT, WIDTH))

while True:
    # Extract and parse data
    msg = df['data'][frame_index]
    index = msg.find("data=")
    array_start = index + 6
    occupancy_str = msg[array_start:-2]
    values = [int(x) for x in occupancy_str.split(',')]
    occupancy_arr = np.array(values).reshape(HEIGHT, WIDTH)

    # BUFFER
    buffered_binary = ndimage.binary_dilation(occupancy_arr, iterations=1).astype(int)
    buffered_binary_conv, Confidence_values_conv = Filtered_Occupancy_Convolution_Masked(buffered_binary, C_old_conv)

    visibility_t1 = (buffered_binary_conv > 0)
        
    # Optical Flow
    masked_old_confidence = np.where(visibility_t1, C_old_conv, 0).astype(np.float32)
    masked_curr_confidence  = np.where(visibility_t1, Confidence_values_conv, 0).astype(np.float32)
    flow = cv2.calcOpticalFlowFarneback(masked_old_confidence, masked_curr_confidence, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Averaging with square filter
    kernel_size = 5
    vx = uniform_filter(flow[::, ::, 1], size=kernel_size)
    vy = uniform_filter(flow[::, ::, 1], size=kernel_size)
    # Scale up for visibility
    fx = vx[::step,::step]*scale_factor
    fy = vy[::step, ::step]*scale_factor

    binary_map, T_range = thresholding_std(C_old_conv, Confidence_values_conv)

    # Update plots
    im0.set_data(buffered_binary_conv)
    axs[0].set_title(f"Convolution Frame\nFrame {frame_index}")
    im1.set_data(Confidence_values_conv)
    axs[1].set_title(f"Confidence from Convolution on \nframe {frame_index}")
    im2.set_data(C_old_conv)
    axs[2].set_title(f"Confidence from Convolution on Previous \nframe {frame_index-1}")
    im3.set_data(Confidence_values_conv)
    quiver.set_offsets(np.stack((x_quiv.ravel(), y_quiv.ravel()), axis=-1))
    quiver.set_UVC(fx.ravel(), fy.ravel())
    axs[3].set_title("Applied optical flow")
    im4.set_data(binary_map)
    axs[4].set_title("Binary Boundary Map")

    fig.canvas.draw_idle()
    frame_index += 1
    plt.pause(0.02)

    # Update previous
    prev_buffered_binary = buffered_binary.copy()
    C_old_conv = Confidence_values_conv

# Bind the handler
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
