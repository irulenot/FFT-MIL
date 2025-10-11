import pandas as pd
import h5py
import numpy as np

# Load the CSV file
csv_file_path = '../CLAM/BRACS_list_CLAM.csv'
df = pd.read_csv(csv_file_path)

# BRACS
# Path to the HDF5 file
h5_path = '/data/fftmil/BRACS/features_CLAM/h5_files/features_CLAM.h5'

with h5py.File(h5_path, 'w') as h5f:
    for index, row in df.iterrows():
        # Create a group for each slide_id
        group = h5f.create_group(row['slide_id'])
        
        # Store label as an attribute
        group.attrs['label'] = row['label']
        group.attrs['feats'] = '/data/fftmil/BRACS/features_CLAM/pt_files/' + row['slide_id'] + '.pt'

print("HDF5 file created and data stored successfully.")

# IMP
# Load the CSV file
csv_file_path = '../CLAM/IMP_list_CLAM.csv'
df = pd.read_csv(csv_file_path)

# Path to the HDF5 file
h5_path = '/data/fftmil/IMP/features_CLAM/h5_files/features_CLAM.h5'

with h5py.File(h5_path, 'w') as h5f:
    for index, row in df.iterrows():
        # Create a group for each slide_id
        group = h5f.create_group(row['slide_id'])
        
        # Store label as an attribute
        group.attrs['label'] = row['label']
        group.attrs['feats'] = '/data/fftmil/IMP/features_CLAM/pt_files/' + row['slide_id'] + '.pt'

print("HDF5 file created and data stored successfully.")

# LUAD
# Load the CSV file
csv_file_path = '../CLAM/LUAD_list_CLAM.csv'
df = pd.read_csv(csv_file_path)

# Path to the HDF5 file
h5_path = '/data/fftmil/LUAD/features_CLAM/h5_files/features_CLAM.h5'

with h5py.File(h5_path, 'w') as h5f:
    for index, row in df.iterrows():
        # Create a group for each slide_id
        group = h5f.create_group(row['slide_id'])
        
        # Store label as an attribute
        group.attrs['label'] = row['label']
        group.attrs['feats'] = '/data/fftmil/LUAD/features_CLAM/pt_files/' + row['slide_id'] + '.pt'

print("HDF5 file created and data stored successfully.")