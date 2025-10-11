import os

# Define the desired directory structure
base_dir = "/data/fftmil/"
sub_dirs = ["BRACS", "LUAD", "IMP"]

# Create base directory if it doesn't exist
os.makedirs(base_dir, exist_ok=True)

# Create each subdirectory and its "images" subdirectory
for sub in sub_dirs:
    main_path = os.path.join(base_dir, sub)
    images_path = os.path.join(main_path, "images")

    os.makedirs(images_path, exist_ok=True)
    print(f"Created directory: {main_path}")
    print(f"Created directory: {images_path}")
