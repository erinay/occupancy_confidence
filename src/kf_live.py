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
df = pd.read_csv("data/mEnv_mRobf/_map_convert.csv")

HEIGHT, WIDTH, N_frames = 120, 120, 5
step=1
scale_factor=5e4
# scale_factor=1e8*2.5
y_quiv, x_quiv = np.mgrid[0:HEIGHT:step, 0:WIDTH:step]

fig, axs = plt.subplots(2,4, figsize=(15,8))
axs = axs.flatten()

im0 = axs[0].imshow(np.zeros((HEIGHT,WIDTH)), cmap='gray', vmin=0, vmax=1,  origin='lower') # incoming data
im1 = axs[1].imshow(np.zeros((HEIGHT,WIDTH)), cmap='viridis', vmin=0, vmax=3,  origin='lower') # overlay velocity from "raw"
im2 = axs[2].imshow(np.zeros((HEIGHT,WIDTH)), cmap='viridis', vmin=0, vmax=3,  origin='lower') # predict next map with velocity data
im3 = axs[3].imshow(np.zeros((HEIGHT,WIDTH)), cmap='viridis', vmin=0, vmax=3,  origin='lower') # combine

im4 = axs[4].imshow(np.zeros((HEIGHT,WIDTH)), cmap='viridis', vmin=0, vmax=10,  origin='lower') # convolution
im5 = axs[5].imshow(np.zeros((HEIGHT,WIDTH)), cmap='viridis', vmin=0, vmax=3,  origin='lower') # overlay velocity from "convolution"
im6 = axs[6].imshow(np.zeros((HEIGHT,WIDTH)), cmap='viridis', vmin=0, vmax=3,  origin='lower') # predict next map
im7 = axs[7].imshow(np.zeros((HEIGHT,WIDTH)), cmap='viridis', vmin=0, vmax=1,  origin='lower') #  combine


# Define quiver
quiver = axs[3].quiver(x_quiv, y_quiv, np.zeros_like(x_quiv), np.zeros_like(y_quiv), 
                       color='red', scale=1, scale_units='xy', angles='xy')
quiver2 = axs[4].quiver(x_quiv, y_quiv, np.zeros_like(x_quiv), np.zeros_like(y_quiv), 
                       color='red', scale=1, scale_units='xy', angles='xy')

for ax in axs:
    ax.axis("off")

# Global Variables
frame_init=1000
frame_index=frame_init
BUFFERED_BINARY_FRAMES=[]
prev_buffered_binary = np.zeros((HEIGHT,WIDTH))
prev_convolution = np.zeros((HEIGHT,WIDTH))

# necessary for current function definitions (i think?, i.e. not necessarily used)
Confidence_values_conv = np.zeros((HEIGHT, WIDTH))
C_old_conv = np.zeros((HEIGHT, WIDTH))  # Initialize confidence values for convolution


while(True):
    # Extract and parse data
    msg = df['data'][frame_index]
    index = msg.find("data=")
    array_start = index + 6
    occupancy_str = msg[array_start:-2]
    values = [int(x) for x in occupancy_str.split(',')]
    occupancy_arr = np.array(values).reshape(HEIGHT, WIDTH)

    # BUFFER & 
    buffered_binary = ndimage.binary_dilation(occupancy_arr, iterations=1).astype(int)
    BUFFERED_BINARY_FRAMES.append(buffered_binary)
    if len(BUFFERED_BINARY_FRAMES) > N_frames:
        BUFFERED_BINARY_FRAMES.pop(0)

    buffered_binary_conv, Confidence_values_conv, _ = Filtered_Occupancy_Convolution(BUFFERED_BINARY_FRAMES, C_old_conv)

    visibility_t1 = (buffered_binary_conv > 0)
        
    # Optical Flow
    masked_old_confidence = np.where(visibility_t1, C_old_conv, 0).astype(np.float32)
    masked_curr_confidence  = np.where(visibility_t1, Confidence_values_conv, 0).astype(np.float32)
    flow = cv2.calcOpticalFlowFarneback(masked_old_confidence, masked_curr_confidence, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # # Try gaussian Filter
    # vx = gaussian_filter(flow[::, ::, 0], sigma=1)
    # vy = gaussian_filter(flow[::, ::, 1], sigma=1)
    # Averaging with square filter
    kernel_size = 5
    vx = uniform_filter(flow[::, ::, 1], size=kernel_size)
    vy = uniform_filter(flow[::, ::, 1], size=kernel_size)
    # Scale up for visibility
    fx = vx[::step,::step]*scale_factor
    fy = vy[::step, ::step]*scale_factor

    speed = np.hypot(vx, vy)
    print("Confidence map min/max:", Confidence_values_conv.min(), Confidence_values_conv.max())    # 2) Connected components labeling
    print("Speed map min/max:", speed.min(), speed.max())    # 2) Connected components labeling

    # Cluster 
    labels, blobs = get_motion_blobs(vx, vy, Confidence_values_conv)
    U = np.zeros_like(vx)  # shape (rows, cols)
    V = np.zeros_like(vy)

    for b in blobs:
        r = int(b['centroid'][0])
        c = int(b['centroid'][1])
        U[r, c] = b['avg_vx']*scale_factor
        V[r, c] = b['avg_vy']*scale_factor

    quiver2.set_UVC(U, V)

    # Evolve map (Option 1: evolve map; Option2: )
    predicted_map = evolve(Confidence_values_conv, labels, blobs, 10000)
    belief_map = filter(predicted_map, Confidence_values_conv, alpha=0.2)

    # Update plots
    im0.set_data(buffered_binary)
    axs[0].set_title(f"Buffered Binary Frame\nFrame {frame_index}")
    im1.set_data(Confidence_values_conv)
    axs[1].set_title(f"Confidence from Convolution on \nframe {frame_index}")
    im2.set_data(C_old_conv)
    axs[2].set_title(f"Confidence from Convolution on Previous \nframe {frame_index-1}")
    im3.set_data(Confidence_values_conv)
    quiver.set_offsets(np.stack((x_quiv.ravel(), y_quiv.ravel()), axis=-1))
    quiver.set_UVC(fx.ravel(), fy.ravel())
    axs[3].set_title("Applied optical flow")

    im4.set_data(labels)
    # Suppose labels is your cluster label map (0 = background)
    num_clusters = labels.max()
    im4.set_clim(0, max(num_clusters, 1))  
    axs[4].set_title(f"Clusters and their velocities in Frame {frame_index}\n *velocity vector scaled up for ease of viewing")
    axs[5].imshow(predicted_map, cmap='viridis', origin='lower')
    axs[5].imshow(Confidence_values_conv, cmap='Reds', origin='lower', alpha=0.4)  # translucent overlay
    im6.set_data(belief_map)
    axs[6].set_title(f"Combine predicted and actual map belief")

    fig.canvas.draw_idle()
    
    frame_index += 1

    # Update previous
    prev_buffered_binary = buffered_binary.copy()
    C_old_conv = belief_map
    plt.pause(0.05)
