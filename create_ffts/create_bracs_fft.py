import os
import numpy as np
import torch
from tqdm import tqdm
from monai.transforms import LoadImaged
from monai.data.wsi_reader import WSIReader
import torch.fft
import random
import torchvision.transforms as T
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

input_dir = '/data/fftmil/BRACS/images/'
output_dir = '/data/fftmil/BRACS/bracs_fft/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_tiff(tiff_file):
    tiff_file0 = tiff_file
    tiff_file = tiff_file.split('/')[-1]
    output_file = os.path.join(output_dir, tiff_file)[:-4]
    if os.path.exists(f'{output_file}.npz'):
        return None  # Skip if already processed

    loader = LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=1, image_only=True)
    image_dict = loader({"image": tiff_file0})
    x = image_dict["image"]

    tensor = x
    # Assuming 'tensor' is your RGB image tensor with shape [3, 2128, 4200]
    # Apply 2D FFT across each color channel separately
    tensor = torch.fft.fft2(tensor, dim=(-2, -1))
    # If you need to shift the zero-frequency component to the center of the spectrum
    tensor = torch.fft.fftshift(tensor, dim=(-2, -1))
    magnitude = torch.abs(tensor)
    phase = torch.angle(tensor)

    tensor = magnitude
    # Target size
    target_height = 2048
    target_width = 2048
    # Step 1: Determine cropping
    # Calculate the start indices for cropping (to crop from the center)
    start_x = max(0, (tensor.size(2) - target_width) // 2)
    start_y = max(0, (tensor.size(1) - target_height) // 2)
    # End indices for cropping
    end_x = start_x + min(target_width, tensor.size(2))
    end_y = start_y + min(target_height, tensor.size(1))
    # Perform cropping
    cropped_tensor = tensor[:, start_y:end_y, start_x:end_x]
    # Step 2: Determine padding needed after cropping
    pad_height = max(0, target_height - cropped_tensor.size(1)) // 2
    pad_width = max(0, target_width - cropped_tensor.size(2)) // 2
    # Apply symmetric padding if necessary
    padded_tensor = F.pad(cropped_tensor, (pad_width, pad_width, pad_height, pad_height), "constant", 0)
    # If the original dimensions are odd, you might need to add an extra padding to one side
    extra_height = target_height - padded_tensor.size(1)
    extra_width = target_width - padded_tensor.size(2)
    magnitude = F.pad(padded_tensor, (0, extra_width, 0, extra_height), "constant", 0)

    tensor = phase
    # Target size
    target_height = 2048
    target_width = 2048
    # Step 1: Determine cropping
    # Calculate the start indices for cropping (to crop from the center)
    start_x = max(0, (tensor.size(2) - target_width) // 2)
    start_y = max(0, (tensor.size(1) - target_height) // 2)
    # End indices for cropping
    end_x = start_x + min(target_width, tensor.size(2))
    end_y = start_y + min(target_height, tensor.size(1))
    # Perform cropping
    cropped_tensor = tensor[:, start_y:end_y, start_x:end_x]
    # Step 2: Determine padding needed after cropping
    pad_height = max(0, target_height - cropped_tensor.size(1)) // 2
    pad_width = max(0, target_width - cropped_tensor.size(2)) // 2
    # Apply symmetric padding if necessary
    padded_tensor = F.pad(cropped_tensor, (pad_width, pad_width, pad_height, pad_height), "constant", 0)
    # If the original dimensions are odd, you might need to add an extra padding to one side
    extra_height = target_height - padded_tensor.size(1)
    extra_width = target_width - padded_tensor.size(2)
    phase = F.pad(padded_tensor, (0, extra_width, 0, extra_height), "constant", 0)

    x = torch.cat([magnitude, phase], dim=0) 

    np.savez_compressed(f'{output_file}', x.numpy())  # Use compressed format
    return output_file

# Loop until all files are processed
while True:
    # Get list of unprocessed TIFF files
    # List to store .svs file paths
    tiff_files = []

    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.svs'):
                # Add full file path to the list
                tiff_files.append(os.path.join(root, file))
    
    existing_npz_files = {os.path.splitext(file)[0] for file in os.listdir(output_dir) if file.endswith('.npz')}
    tiff_files = [
    svs_file for svs_file in tiff_files
    if os.path.splitext(os.path.basename(svs_file))[0] not in existing_npz_files
    ]
    
    remaining_files = len(tiff_files)  # Dynamically calculate remaining files

    # Print status of remaining files
    print(f'Remaining files: {remaining_files}')

    if not tiff_files:
        print('All TIFF files have been processed.')
        break  # Exit the loop when no unprocessed files are found

    random_tiff_file = random.choice(tiff_files)  # Select a random unprocessed file
    result = process_tiff(random_tiff_file)
    
    if result:
        print(f'Processed and saved: {result}')
    else:
        print(f'{random_tiff_file} was already processed.')
