import os
import numpy as np
import pandas as pd
import shutil
import re
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from IPython.display import display
from scipy.ndimage import binary_dilation, gaussian_filter, label

def parse_index_ranges(range_str):
    indices = set()
    for part in range_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return indices

def select_it(input_dir, output_dir, index_range_str, ignore_files=None):
    """
    Copies .png files from 'sprites/final' subdirectories and from a 'baseline' folder,
    if their numeric index is in the given range string.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    ignore_files = ignore_files or set()
    selected_indices = parse_index_ranges(index_range_str)

    # Copy baseline folder (filtered by range)
    baseline_input = input_dir / "baseline"
    if baseline_input.exists():
        baseline_output = output_dir / "baseline"
        baseline_output.mkdir(parents=True, exist_ok=True)
        for file in baseline_input.iterdir():
            if file.suffix == ".png":
                try:
                    file_index = int(file.stem)
                    if file_index in selected_indices:
                        shutil.copy2(file, baseline_output / file.name)
                except ValueError:
                    continue

    # Copy folders with sprites
    for folder in input_dir.iterdir():
        if not folder.is_dir() or folder.name == "baseline":
            continue

        sprites_final_path = folder / "sprites" / "final"
        if sprites_final_path.exists():
            new_folder_path = output_dir / folder.name
            new_folder_path.mkdir(parents=True, exist_ok=True)

            for file in sprites_final_path.iterdir():
                if file.suffix == '.png' and file.name not in ignore_files:
                    try:
                        file_index = int(file.stem)
                        if file_index in selected_indices:
                            destination = new_folder_path / file.name
                            shutil.copy2(file, destination)
                            try:
                                # Verify PNG integrity
                                with Image.open(destination) as img:
                                    img.verify()
                            except Exception as e:
                                print(f"[Warning] Invalid PNG: {destination.name} in {folder.name} - {e}")
                                destination.unlink(missing_ok=True)
                    except ValueError:
                        continue
                    
def crop_it(input_dir, output_dir,
                              crop_top=None, crop_bottom=None,
                              crop_left=None, crop_right=None,
                              extensions=('.png', '.jpg', '.jpeg')):
    """
    Crops pixels from the edges of all images in a directory (recursively),
    and saves the cropped images in a mirrored output directory structure.

    Parameters:
    - input_dir (str): Path to the input directory containing images.
    - output_dir (str): Path to the output directory to save cropped images.
    - crop_top, crop_bottom, crop_left, crop_right (int): Pixels to crop from each side.
    - extensions (tuple): Allowed image file extensions.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(extensions):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, os.path.relpath(input_path, input_dir))

                # Create output directory structure if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with Image.open(input_path) as image:
                    width, height = image.size

                    # Ensure crop values are within image bounds
                    if (crop_left + crop_right < width) and (crop_top + crop_bottom < height):
                        cropped_image = image.crop((
                            crop_left,
                            crop_top,
                            width - crop_right,
                            height - crop_bottom
                        ))
                        cropped_image.save(output_path)
                    else:
                        print(f"Skipping {input_path}: Crop dimensions exceed image size.")

def filter_it(input_dir, output_dir, reference_dir, index_range=None, threshold_ratio=None):
    """
    Filters a batch of image folders by applying masks derived from a reference folder.
    Overwrites existing files in the output directory if they already exist.

    Parameters:
    - input_dir (str): Path to folders containing input images.
    - output_dir (str): Path where filtered images will be saved.
    - reference_dir (str): Path to reference masks (e.g., baseline predictions).
    - index_range (range or None): Index range of image filenames to process. If None, process all PNGs.
    - threshold_ratio (float): Threshold ratio for binary mask creation.
    """
    import os
    import numpy as np
    from PIL import Image
    from scipy.ndimage import binary_dilation, label, gaussian_filter
    
    for folder in os.listdir(input_dir):
        if folder.startswith('.') or folder == 'baseline':
            continue  # Skip hidden/system folders and the reference folder itself
        
        folder_input = os.path.join(input_dir, folder)
        folder_output = os.path.join(output_dir, folder)

        # Create output directory if it doesn't exist (instead of failing if it exists)
        os.makedirs(folder_output, exist_ok=True)

        images = []
        masks_ref = []
        valid_filenames = []

        # Determine filenames to process
        if index_range is None:
            # Take all PNG files in folder_input
            filenames = [f for f in os.listdir(folder_input) if f.endswith('.png')]
        else:
            # Build filenames from index_range
            filenames = [f"{i}.png" for i in index_range]

        for filename in filenames:
            image_path = os.path.join(folder_input, filename)
            mask_path = os.path.join(reference_dir, filename)

            try:
                image = Image.open(image_path).convert("L")
                mask = Image.open(mask_path).convert("L")
            except FileNotFoundError:
                print(f"Missing image or mask: {filename} — skipping.")
                continue

            mask_array = np.array(mask)
            threshold_value = threshold_ratio * 255
            binary_mask = np.where(mask_array < threshold_value, 255, 0).astype(np.uint8)
            dilated_mask = binary_dilation(binary_mask)
            dilated_mask = binary_dilation(dilated_mask)

            labeled_mask, num_features = label(dilated_mask)
            if num_features == 0:
                largest_component_mask = binary_mask
            else:
                component_sizes = [np.sum(labeled_mask == idx) for idx in range(1, num_features + 1)]
                largest_label = np.argmax(component_sizes) + 1
                largest_component_mask = np.where(labeled_mask == largest_label, 255, 0).astype(np.uint8)

            convolved_mask = gaussian_filter(largest_component_mask, sigma=1)

            images.append(image)
            masks_ref.append(convolved_mask)
            valid_filenames.append(filename)

        # Apply masks and save filtered images
        for image, mask, filename in zip(images, masks_ref, valid_filenames):
            img_array = np.array(image)
            filtered_image = 255 - mask + img_array * (mask / 255)
            filtered_image = Image.fromarray(filtered_image.astype(np.uint8))
            
            # Will overwrite if the file already exists
            output_path = os.path.join(folder_output, filename)
            filtered_image.save(output_path)
            print(f"Saved filtered image to: {output_path}")



def flag_it(input_dir, output_dir, reference_dir,
            index_range=None,
            default_threshold_ratio=None, sup_threshold_ratio=None,
            orange_threshold=None, red_threshold=None):
    """
    Flags differences between reference masks and input images by applying visual borders and diagnostics.

    Parameters:
    - input_dir (Path or str): Folder containing input folders with grayscale images.
    - output_dir (Path or str): Folder where output images (with borders and diagnostics) are saved.
    - reference_dir (Path or str): Folder containing reference masks (directly, not in subfolders).
    - index_range (range or list): Range or list of image indices to process.
    - default_threshold_ratio (float): Threshold ratio for binarizing reference masks.
    - sup_threshold_ratio (float): Threshold ratio for binarizing input images for "sup" masks.
    - orange_threshold (tuple): (min, max) range to assign orange border.
    - red_threshold (float): Minimum difference to assign red border.

    Returns:
    - pd.DataFrame: DataFrame with folder names, image indices, and difference scores.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    from scipy.ndimage import binary_dilation, label, gaussian_filter
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    reference_dir = Path(reference_dir)

    default_threshold = default_threshold_ratio * 255
    sup_threshold = sup_threshold_ratio * 255

    difference_data = []

    for folder in input_dir.iterdir():
        if folder.name.startswith('.') or not folder.is_dir():
            continue

        folder_output = output_dir / folder.name
        folder_output.mkdir(parents=True, exist_ok=True)

        # Determine filenames to process
        if index_range is None:
            filenames = sorted([f.name for f in folder.glob("*.png")])
        else:
            filenames = [f"{i}.png" for i in index_range]

        for filename in filenames:
            i = Path(filename).stem
            image_path = folder / filename
            
            # *** Key change: The mask is now directly in reference_dir, not in a subfolder ***
            mask_path = reference_dir / filename

            if not image_path.exists() or not mask_path.exists():
                print(f"Looking for image at: {image_path}")
                print(f"Looking for mask at: {mask_path}")
                print(f"Missing image or mask: {filename} in folder {folder.name} — skipping.")
                continue

            image = Image.open(image_path).convert("L")
            mask = Image.open(mask_path).convert("L")

            img_array = np.array(image)
            mask_array = np.array(mask)

            # Process reference mask
            binary_mask_ref = np.where(mask_array < default_threshold, 255, 0).astype(np.uint8)

            dilated_mask_ref = binary_mask_ref
            for _ in range(2):
                dilated_mask_ref = binary_dilation(dilated_mask_ref)

            labeled_ref, n_ref = label(dilated_mask_ref)
            if n_ref > 0:
                largest_label_ref = np.argmax([np.sum(labeled_ref == l) for l in range(1, n_ref + 1)]) + 1
                largest_mask_ref = (labeled_ref == largest_label_ref).astype(np.uint8) * 255
            else:
                largest_mask_ref = dilated_mask_ref

            convolved_mask_ref = gaussian_filter(largest_mask_ref, sigma=1)

            # Process sup mask from input image
            binary_mask_sup = np.where(img_array < sup_threshold, 255, 0).astype(np.uint8)
            labeled_sup, n_sup = label(binary_mask_sup)
            if n_sup > 0:
                largest_label_sup = np.argmax([np.sum(labeled_sup == l) for l in range(1, n_sup + 1)]) + 1
                largest_mask_sup = (labeled_sup == largest_label_sup).astype(np.uint8) * 255
            else:
                largest_mask_sup = binary_mask_sup

            # Create filtered image based on reference mask
            filtered_ref = 255 - convolved_mask_ref + img_array * (convolved_mask_ref / 255)
            filtered_ref = np.clip(filtered_ref, 0, 255).astype(np.uint8)
            filtered_ref_img = Image.fromarray(filtered_ref).convert("RGB")

            # Calculate difference score
            difference = (np.sum(largest_mask_sup * (255. - convolved_mask_ref)) / 255.) / 255

            # Determine border color
            if orange_threshold[0] <= difference <= orange_threshold[1]:
                border_color = 'orange'
            elif difference > red_threshold:
                border_color = 'red'
            else:
                border_color = 'white'

            # Add border and save
            filtered_with_border = ImageOps.expand(filtered_ref_img, border=2, fill=border_color)
            filtered_with_border.save(folder_output / f"{i}.png")

            # Save diagnostic figure
            difference_array = (largest_mask_sup * (255. - convolved_mask_ref)) / 255.
            difference_image = Image.fromarray(difference_array.astype(np.uint8))

            fig, axes = plt.subplots(1, 5, figsize=(12, 4))
            axes[0].imshow(img_array, cmap='gray')
            axes[0].set_title(f'Original ({folder.name})')
            axes[1].imshow(convolved_mask_ref, cmap='gray')
            axes[1].set_title('Ref Mask')
            axes[2].imshow(largest_mask_sup, cmap='gray')
            axes[2].set_title('Sup Mask')
            axes[3].imshow(filtered_ref, cmap='gray')
            axes[3].set_title('Filtered Ref')
            axes[4].imshow(difference_image, cmap='gray')
            axes[4].set_title(f'Diff: {difference:.2f}')
            for ax in axes:
                ax.axis('off')

            fig.savefig(folder_output / f"difference_map_{i}.png", bbox_inches='tight')
            plt.close(fig)

            difference_data.append({'Folder': folder.name, 'Image': i, 'Difference': difference})

    return pd.DataFrame(difference_data)

def make_prototype_grid(
    folder_paths,
    output_path=None,
    margin=2,
    show=True
):
    """
    Creates and optionally displays a grid of prototype images arranged by folder and image IDs.
    Only collects images with integer filenames (e.g., "26.png", "49.png").
    Sorts folder names by numeric part after '_f' (handles suffix letters),
    and normalizes '_f960' to '_f96' before sorting.

    Args:
        folder_paths (str or Path): Root directory where all folder subpaths exist.
        output_path (str or Path, optional): Where to save the final grid. If None, does not save.
        margin (int): Pixel space between images.
        show (bool): If True, display the final grid in Jupyter.
    """
    import re
    from pathlib import Path
    from PIL import Image
    from IPython.display import display
    
    folder_paths = Path(folder_paths)
    folders = [f.name for f in folder_paths.iterdir() if f.is_dir()]

    # Adjust folder names for sorting (internally rename '_f960' -> '_f96')
    def sort_key(folder_name):
        # Replace '_f960' with '_f96'
        adjusted = folder_name.replace('_f960', '_f96')

        # Extract numeric part after '_f' (and optional letter)
        # e.g. btv1b84472995_f57a -> 57, 'a'
        match = re.search(r'_f(\d+)([a-z]?)$', adjusted)
        if match:
            number = int(match.group(1))
            letter = match.group(2)
            # letter to ASCII for secondary sorting, empty letter comes first
            letter_val = ord(letter) if letter else 0
            return (number, letter_val)
        else:
            # fallback: put it at the end
            return (9999, 0)

    folders.sort(key=sort_key)

    print("Sorted folders:", folders)

    # Automatically get all image ids from first folder (assuming consistent naming)
    # MODIFIED: Only select filenames that are strictly integers
    first_folder = folder_paths / folders[0]
    all_png_files = first_folder.glob("*.png")
    
    # Filter to keep only filenames that are strictly integers
    image_ids = []
    for png_file in all_png_files:
        stem = png_file.stem
        if stem.isdigit():  # Check if the stem contains only digits
            image_ids.append(stem)
    
    # Sort numerically (not alphabetically)
    image_ids = sorted(image_ids, key=int)
    
    print(f"Found {len(image_ids)} integer-named images in {folders[0]}")

    def horzpil(images, width, height, margin=2):
        images = [Image.open(image_path) for image_path in images]
        total_width = width * len(images) + margin * (len(images) - 1)
        new_im = Image.new('RGB', (total_width, height), color=(255, 255, 255))

        x_offset = 0
        for img in images:
            new_im.paste(img, (x_offset, 0))
            x_offset += width + margin
        return new_im

    # Read one image to determine dimensions
    if not image_ids:
        print("No integer-named images found.")
        return
        
    sample_img_path = folder_paths / folders[0] / f"{image_ids[0]}.png"
    sample_img = Image.open(sample_img_path)
    img_w, img_h = sample_img.size

    grid_images = []
    for folder_name in folders:
        row_images = []
        folder_dir = folder_paths / folder_name
        for img_id in image_ids:
            img_path = folder_dir / f"{img_id}.png"
            if img_path.exists():
                row_images.append(img_path)
        if row_images:
            grid_row = horzpil(row_images, img_w, img_h, margin)
            grid_images.append(grid_row)

    if not grid_images:
        print("No images found.")
        return

    # Combine rows vertically
    total_height = sum(img.height for img in grid_images)
    final_img = Image.new('RGB', (grid_images[0].width, total_height), color=(255, 255, 255))

    y_offset = 0
    for row in grid_images:
        final_img.paste(row, (0, y_offset))
        y_offset += row.height

    if output_path:
        final_img.save(output_path)
        print(f"Grid image saved to: {output_path}")

    if show:
        display(final_img)

    #return final_img
