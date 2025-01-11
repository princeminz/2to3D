import torch
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as T
from moge.model import MoGeModel
from time import time
import cv2
import numpy as np
from time import time
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-bs', '--batchsize')
parser.add_argument('-o', '--output', default="side_by_side_video.mp4")

args = parser.parse_args()

FNAME = args.input
BATCH_SIZE = int(args.batchsize)
OUTFILE = args.output

# Get user input for video clipping
start_frame = int(input("Enter the starting frame number: "))
total_frames = int(input("Enter the total number of frames for the clip: "))

# Calculate the number of batches to process
batches_to_process = (total_frames + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
frames_processed = 0
left_outs = []
right_outs = []
total_time = 0
save_per_x_batch = 1
save_chunks_dir = os.path.join(os.getcwd(), ".chunks")
os.makedirs(save_chunks_dir, exist_ok=True)

def create_side_by_side_video(left_batch, right_batch, out):
    for left, right in zip(left_batch, right_batch):
        # Convert torch tensors to numpy arrays and then to uint8 for cv2
        left_frame = (left.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        right_frame = (right.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)

        # Combine left and right frames side by side
        combined_frame = np.hstack((left_frame, right_frame))
        
        # Write the frame into the video
        out.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))  # OpenCV uses BGR, convert from RGB



def create_stereo_images(image, depth_map, divergence=2, convergence=0.5):
    """
    Generate stereo 3D images using bilinear interpolation via grid_sample.

    :param image: Tensor of shape [B, C, H, W], the input image.
    :param depth_map: Tensor of shape [B, H, W], the depth information.
    :param divergence: Controls the stereo divergence effect.
    :param convergence: Adjusts how depth affects pixel displacement.
    :return: Tuple of tensors for left and right eye views.
    """
    batch_size, channels, height, width = image.shape
    
    # Calculate padding
    padding = calculate_padding(width, divergence)
    padder = torch.nn.ReplicationPad2d((padding, padding, 0, 0))
    padded_image = padder(image)
    padded_depth = padder(depth_map)
    
    # Calculate pixel shift based on depth
    shift_size = width * 0.01 * divergence / 2
    shift = padded_depth * shift_size - (shift_size * convergence)
    
    # Create stereo views - normalized grid for grid_sample
    left_grid = create_grid(shift, height, width + 2*padding, batch_size)
    right_grid = create_grid(-shift, height, width + 2*padding, batch_size)
    
    left_eye = F.grid_sample(padded_image, left_grid, mode='bilinear', padding_mode='border', align_corners=True)
    right_eye = F.grid_sample(padded_image, right_grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    # Remove padding
    unpadder = torch.nn.ReplicationPad2d((-padding, -padding, 0, 0))
    return unpadder(left_eye), unpadder(right_eye)

def calculate_padding(width, divergence):
    """Calculate padding size based on image width and divergence."""
    return int(width * divergence * 0.01 + 2)

def create_grid(shift, height, width, batch_size):
    """
    Create a grid for grid_sample based on the shift.

    :param shift: Shift tensor based on depth.
    :param height: Image height.
    :param width: Image width.
    :param batch_size: Number of images in the batch.
    :return: Grid tensor for grid_sample.
    """
    y = torch.linspace(-1, 1, height, device=shift.device).view(1, height, 1).expand(batch_size, height, width)
    x = torch.linspace(-1, 1, width, device=shift.device).view(1, 1, width).expand(batch_size, height, width)
    
    # Normalize shift to [-1, 1] range
    shift_normalized = shift * 2 / (width - 1)
    
    # Combine into a grid
    grid = torch.stack([x + shift_normalized, y], dim=3)
    return grid

mem_used_mb_pb = 0

def process_batch(batch, model, device):
    """
    Process a single batch through the model and generate stereo images.

    :param batch: A batch of video frames.
    :param model: The MoGe model for depth estimation.
    :param device: The device to run computations on.
    :return: Tuple of left and right eye images.
    """
    global mem_used_mb_pb
    free, total = torch.cuda.mem_get_info(device)
    mem_used_mb_pb = (total - free) / 1024 ** 2

    # Normalize input to [0, 1] range
    inp = (batch[0]).to(device) / 255.0
    
    try:
        # Model inference
        free, total = torch.cuda.mem_get_info(device)
        mem_used_mb = (total - free) / 1024 ** 2
        out = model.infer(inp)
        free, total = torch.cuda.mem_get_info(device)
        mem_used_mb_a = (total - free) / 1024 ** 2
        if mem_used_mb_a > mem_used_mb: print("After Model Infer:", mem_used_mb, mem_used_mb_a)
        
        # Extract depth, considering only valid areas (mask > 0)
        valid_mask = out['mask'] > 0
        valid_depth = out['depth'][valid_mask]
        
        # Normalize depth to [0, 1] for consistency with stereo creation
        depth_min = valid_depth.amin()
        depth_max = valid_depth.amax()
        normalized_depth = torch.zeros_like(out['depth'])
        normalized_depth[valid_mask] = 1. - ((valid_depth - depth_min) / (depth_max - depth_min))
        # free, total = torch.cuda.mem_get_info(device)
        # mem_used_mb = (total - free) / 1024 ** 2
        # print("Process Batch Before Create Stereo:", mem_used_mb)

        # Create stereo images
        left_out, right_out = create_stereo_images(inp, normalized_depth)
        
        # free, total = torch.cuda.mem_get_info(device)
        # mem_used_mb = (total - free) / 1024 ** 2
        # print("Process Batch After Create Sterio:", mem_used_mb)

        return left_out, right_out
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return None, None
    

# Main execution loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
stream = torchaudio.io.StreamReader(FNAME)
stream.add_basic_video_stream(frames_per_chunk=BATCH_SIZE)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
height, width = None, None
out = None
fps = 25

# Process only the specified frames
for i, batch in enumerate(stream.stream()):
    
    if i == 0:
        height, width = batch[0].shape[2:4]
        print(batch[0].shape)
        out = cv2.VideoWriter(OUTFILE, fourcc, fps, (width*2, height))

    if frames_processed >= start_frame + total_frames:
        break  # Stop processing once we've processed the desired number of frames

    if frames_processed >= start_frame:  # Start processing from the specified start frame
        start_time = time()
        
        left_out, right_out = process_batch(batch, model, device)
        
        free, total = torch.cuda.mem_get_info(device)
        mem_used_mb_pb_a = (total - free) / 1024 ** 2
        if mem_used_mb_pb_a > mem_used_mb_pb: print("Process Batch End:", mem_used_mb_pb, mem_used_mb_pb_a)

        if left_out is not None and right_out is not None:
            create_side_by_side_video(left_out, right_out, out)
            # left_outs.append(left_out.to('cpu'))
            # right_outs.append(right_out.to('cpu'))
            # print(f"Left out shape: {left_out.shape}, Right out shape: {right_out.shape}")
        else:
            print(f"Batch {i} processing failed")

        batch_time = time() - start_time
        total_time += batch_time
        print(f"Batch {i} Processed in {batch_time:.2f} seconds")
    
    frames_processed += BATCH_SIZE  # Increment by batch size

    # Optional: Clear CUDA cache to free up memory if necessary
    # torch.cuda.empty_cache()

print(f"Total processing time: {total_time:.2f} seconds")

# Create the side-by-side video with only the clipped frames
out.release()
