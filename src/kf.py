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
scale_factor=5e3
# scale_factor=1e8*2.5
y_quiv, x_quiv = np.mgrid[0:HEIGHT:step, 0:WIDTH:step]

fig, axs = plt.subplots(2,4, figsize=(15,8))
axs = axs.flatten()

im0 = axs[0].imshow(np.zeros((HEIGHT,WIDTH)), cmap='gray', vmin=0, vmax=1,  origin='lower') # incoming data
im1 = axs[1].imshow(np.zeros((HEIGHT,WIDTH)), cmap='hot', vmin=0, vmax=3,  origin='lower') # overlay velocity from "raw"
im2 = axs[2].imshow(np.zeros((HEIGHT,WIDTH)), cmap='hot', vmin=0, vmax=3,  origin='lower') # predict next map with velocity data
im3 = axs[3].imshow(np.zeros((HEIGHT,WIDTH)), cmap='hot', vmin=0, vmax=3,  origin='lower') # combine

im4 = axs[4].imshow(np.zeros((HEIGHT,WIDTH)), cmap='viridis', vmin=0, vmax=3,  origin='lower') # convolution
im5 = axs[5].imshow(np.zeros((HEIGHT,WIDTH)), cmap='viridis', vmin=0, vmax=1,  origin='lower') # overlay velocity from "convolution"
im6 = axs[6].imshow(np.zeros((HEIGHT,WIDTH)), cmap='viridis', vmin=0, vmax=1,  origin='lower') # predict next map
im7 = axs[7].imshow(np.zeros((HEIGHT,WIDTH)), cmap='viridis', vmin=0, vmax=1,  origin='lower') #  combine

fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
fig.colorbar(im4, ax=axs[4], fraction=0.046, pad=0.04)


# Define quiver
quiver = axs[3].quiver(x_quiv, y_quiv, np.zeros_like(x_quiv), np.zeros_like(y_quiv), 
                       color='red', scale=1, scale_units='xy', angles='xy')
quiver2 = axs[7].quiver(x_quiv, y_quiv, np.zeros_like(x_quiv), np.zeros_like(y_quiv), 
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
Confidence_values_decay = np.zeros((HEIGHT, WIDTH))  # Initialize confidence values for convolution
confidence_combined = np.zeros((HEIGHT,WIDTH))

def on_key(event):
    global frame_index, BUFFERED_BINARY_FRAMES, prev_buffered_binary, prev_convolution, Confidence_values_conv, Confidence_values_decay, confidence_combined

    if event.key != 'right':
        return

    if frame_index >= len(df):
        print("Reached end of frames.")
        return
    
    # Extract and parse data
    current_time =  df['timestamp'][frame_index]
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

    C_old_conv = Confidence_values_conv
    C_old_comb = confidence_combined

    if len(BUFFERED_BINARY_FRAMES) >= N_frames:
        buffered_binary_conv, Confidence_values_conv, _ = Filtered_Occupancy_Convolution(BUFFERED_BINARY_FRAMES, C_old_conv, N_frames, None)
        status = "ACTIVE"
        confidence_combined = decay_mask_convolution(buffered_binary, C_old_comb)
    else:
        buffered_binary_conv, Confidence_values_conv, _ = Filtered_Occupancy_Convolution(BUFFERED_BINARY_FRAMES, C_old_conv, N_frames, None)
        confidence_combined = decay_mask_convolution(buffered_binary, C_old_comb)
        status = "INACTIVE"

    visibility_t1 = (buffered_binary_conv > 0)
    
    # Create masks to apply optical flow on
    masked_confidence = np.where(visibility_t1, Confidence_values_conv, 0).astype(np.float32)
    
    # Optical Flow
    normed_buffered_conv = buffered_binary_conv/3
    flow = cv2.calcOpticalFlowFarneback(Confidence_values_conv, normed_buffered_conv, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    decay_map = decay_masked(normed_buffered_conv, Confidence_values_decay, visibility_t1, None, decay_rate=0.7)

    # # Try gaussian Filter
    # vx = gaussian_filter(flow[::1, ::1, 0], sigma=1)
    # vy = gaussian_filter(flow[::, ::, 1], sigma=1)
    # Averaging with square filter
    kernel_size = 5
    vx = uniform_filter(flow[::, ::, 1], size=kernel_size)
    vy = uniform_filter(flow[::, ::, 1], size=kernel_size)
    # Scale up for visibility
    fx = vx[::step,::step]*scale_factor
    fy = vy[::step, ::step]*scale_factor

    predicted_map = evolve(buffered_binary, flow)

    # quiver.set_UVC(fx.ravel(), fy.ravel())
    # quiver2.set_UVC(fx.ravel(), fy.ravel())

    speed = np.hypot(vx, vy)
    print("Confidence map min/max:", Confidence_values_conv.min(), Confidence_values_conv.max())    # 2) Connected components labeling
    print("Confidence map min/max:", speed.min(), speed.max())    # 2) Connected components labeling



    # Cluster 
    blobs = get_motion_blobs(vx, vy, Confidence_values_conv)

    if len(blobs) == 0:
        # No motion blobs — clear arrows or set empty data
        quiver2.set_offsets(np.empty((0, 2)))
        # quiver2.set_UVC(np.array([]), np.array([]))
    else:
        x_pos = np.array([b['centroid'][1] for b in blobs])
        y_pos = np.array([b['centroid'][0] for b in blobs])
        vx = np.array([b['avg_vx'] for b in blobs])
        vy = np.array([b['avg_vy'] for b in blobs])

        quiver2.set_offsets(np.stack((x_pos, y_pos), axis=-1))
        quiver2.set_UVC(fx,fy)



    # Update previous
    prev_buffered_binary = buffered_binary.copy()
    Confidence_values_decay = decay_map.copy()

    # Update plots
    # im0.set_data(buffered_binary)
    # axs[0].set_title(f"Buffered Binary Frame\nFrame {frame_index}")
    # im1.set_data(buffered_binary)
    # quiver.set_offsets(np.stack((x_quiv.ravel(), y_quiv.ravel()), axis=-1))
    # quiver.set_UVC(fx.ravel(), fy.ravel())
    # im2.set_data(predicted_map)
    # axs[2].set_title(f"Predicted map at frame {frame_index}")
    # # im3.set_data(filtered_map)
    # im3.set_data(predicted_map)
    # axs[3].set_title(f"Estimated occupancy at frame {frame_index}")

    # Playing with masking for optical flow
    im0.set_data(buffered_binary)
    axs[0].set_title(f"Buffered Binary Frame\nFrame {frame_index}")
    im1.set_data(Confidence_values_conv)
    axs[1].set_title(f"ConfidenceValues from Convolution on frame {frame_index}")
    im2.set_data(masked_confidence)
    axs[2].set_title("Masked Confidence Map")
    im3.set_data(Confidence_values_conv)
    quiver.set_offsets(np.stack((x_quiv.ravel(), y_quiv.ravel()), axis=-1))
    quiver.set_UVC(fx.ravel(), fy.ravel())
    axs[3].set_title("Applied optical flow")

    im7.set_data(Confidence_values_conv)
    # im4.set_data()
    # im4.set_data(buffered_binary_conv)
    # axs[4].set_title(f"Convoluted Binary Frame\nFrame {frame_index}")
    # im5.set_data(buffered_binary_conv)
    # quiver.set_offsets(np.stack((x_quiv.ravel(), y_quiv.ravel()), axis=-1))
    # quiver.set_UVC(fx.ravel(), fy.ravel())
    # im6.set_data(predicted_map)
    # axs[6].set_title(f"Predicted map at frame {frame_index}")
    # im7.set_data(predicted_map)
    # axs[7].set_title(f"Estimated occupancy at frame {frame_index}")
    # print(frame_index)

    fig.canvas.draw_idle()
    frame_index += 1

# Bind the handler
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
