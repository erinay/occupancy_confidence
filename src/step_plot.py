import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
from utils import Filtered_Occupancy, Filtered_Occupancy_Convolution, decay_occupancy
import cv2
import matplotlib.patches as patches


# Load data
df = pd.read_csv("data/_map_convert.csv")

HEIGHT, WIDTH, N_frames = 120, 120, 5
step=2
scale_factor=1e4
y_quiv, x_quiv = np.mgrid[0:HEIGHT:step, 0:WIDTH:step]

# Plot setup
# plt.ion()
fig, axs = plt.subplots(2,3, figsize=(15, 8))
axs = axs.flatten()

im0 = axs[0].imshow(np.zeros((HEIGHT, WIDTH)), cmap='gray', vmin=0, vmax=1, origin='lower')
im1 = axs[1].imshow(np.zeros((HEIGHT, WIDTH)), cmap='hot', vmin=0, vmax=10.0, origin='lower')
im3 = axs[3].imshow(np.zeros((HEIGHT, WIDTH)), cmap='viridis', vmin=0, vmax=3, origin='lower')
im2 = axs[2].imshow(np.zeros((HEIGHT,WIDTH)), origin='lower')
im4 = axs[4].imshow(np.zeros((HEIGHT, WIDTH)), cmap='gray', vmin=0, vmax=1, origin='lower')
im5 = axs[5].imshow(np.zeros((HEIGHT, WIDTH)), origin='lower')

fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

quiver = axs[2].quiver(x_quiv, y_quiv, np.zeros_like(x_quiv), np.zeros_like(y_quiv), 
                       color='red', scale=1, scale_units='xy', angles='xy')

for ax in axs:
    ax.axis("off")

# Globals
frame_index = 0
BUFFERED_BINARY_FRAMES = []
Confidence_values_conv = np.zeros((HEIGHT, WIDTH))
Confidence_values_decay = np.zeros((HEIGHT, WIDTH))  # Initialize confidence values for convolution
prev_filtered_binary_conv = np.zeros((HEIGHT, WIDTH))  
prev_buffered_binary = np.zeros((HEIGHT, WIDTH))

# KEY HANDLER
def on_key(event):
    global frame_index, BUFFERED_BINARY_FRAMES, Confidence_values_conv, Confidence_values_decay, prev_filtered_binary_conv, prev_buffered_binary

    if event.key != 'right':
        return

    if frame_index >= len(df):
        print("Reached end of frames.")
        return

    # Extract and parse data
    msg = df["data"][frame_index]
    index = msg.find("data=")
    array_start = index + 6
    occupancy_str = msg[array_start:-2]
    values = [int(x) for x in occupancy_str.split(',')]
    occupancy_arr = np.array(values).reshape(HEIGHT, WIDTH)

    buffered_binary = ndimage.binary_dilation(occupancy_arr, iterations=1).astype(int)
    # buffered_binary = occupancy_arr
   

    BUFFERED_BINARY_FRAMES.append(buffered_binary)
    if len(BUFFERED_BINARY_FRAMES) > N_frames:
        BUFFERED_BINARY_FRAMES.pop(0)

    C_old_conv = Confidence_values_conv

    if len(BUFFERED_BINARY_FRAMES) >= N_frames:
        Confidence_values, filtered_binary = Filtered_Occupancy(BUFFERED_BINARY_FRAMES, N_frames)
        buffered_binary_conv, Confidence_values_conv, filtered_binary_conv = Filtered_Occupancy_Convolution(BUFFERED_BINARY_FRAMES, C_old_conv, N_frames)
        status = "ACTIVE"
    else:
        Confidence_values = filtered_binary = buffered_binary
        buffered_binary_conv, Confidence_values_conv, filtered_binary_conv = Filtered_Occupancy_Convolution(BUFFERED_BINARY_FRAMES, C_old_conv, N_frames)
        status = "INACTIVE"

    decay_map = decay_occupancy(buffered_binary, Confidence_values_decay, decay_rate=0.9)
    # print(np.amax(decay_map))
    # Draw circles on decay map at pixels with value in [3.0, 3.5]
    circle_radius = 5
    decay_circle_overlay = np.zeros((HEIGHT, WIDTH, 3))  # RGB overlay to draw circles

    ys, xs = np.where((decay_map >= 9.0))
    for x, y in zip(xs, ys):
        cv2.circle(decay_circle_overlay, (x, y), circle_radius, (0.0, 1.0, 0.0), -1)  # red circle

    # Convert decay map to RGB and overlay circles
    decay_rgb = plt.cm.hot(decay_map / 3.5)[..., :3]  # normalize & use colormap (drop alpha)
    decay_rgb = np.clip(decay_rgb + decay_circle_overlay, 0, 1)  # overlay circles

   # Compute change categories
    added = (filtered_binary_conv == 1) & (prev_filtered_binary_conv == 0)
    removed = (filtered_binary_conv == 0) & (prev_filtered_binary_conv == 1)
    unchanged = (filtered_binary_conv == 1) & (prev_filtered_binary_conv == 1)

    # Create RGB image
    change_map = np.zeros((HEIGHT, WIDTH, 3), dtype=float)

    # Assign colors
    change_map[added] = [0, 1, 0]       # Green: new occupied
    change_map[removed] = [1, 0, 0]     # Red: removed
    change_map[unchanged] = [1, 1, 1]   # White: still occupied
    # Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev_buffered_binary, buffered_binary, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Downsample the flow field for quiver
    fx = flow[::step, ::step, 0] * scale_factor
    fy = flow[::step, ::step, 1] * scale_factor

    quiver.set_UVC(fx.ravel(), fy.ravel())


    # Update previous
    prev_filtered_binary_conv = filtered_binary_conv.copy()
    prev_buffered_binary = buffered_binary.copy()

    # Update plots
    im0.set_data(buffered_binary)
    axs[0].set_title(f"Buffered Binary Frame\nFrame {frame_index}")

    im1.set_data(decay_rgb)
    axs[1].set_title(f"Decay map up to {frame_index}")

    im3.set_data(buffered_binary_conv)
    axs[3].set_title(f"Convolution on Buffered Binary Frame\nFrame {frame_index}")

    im2.set_data(buffered_binary)
    quiver.set_offsets(np.stack((x_quiv.ravel(), y_quiv.ravel()), axis=-1))
    quiver.set_UVC(fx.ravel(), fy.ravel())

    im4.set_data(filtered_binary_conv)
    axs[4].set_title("Filtered Occupancy Map\nFiltering: ACTIVE")

    im5.set_data(change_map)
    axs[5].set_title("Change in Filtered Map")

    fig.canvas.draw_idle()
    frame_index += 1

# Bind the handler
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
