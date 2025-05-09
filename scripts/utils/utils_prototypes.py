import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, label

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

def filter_it(input_parent_folder, output_parent_folder, baseline_folder,
                              index_range=None, threshold_ratio=None):
    """
    Filters a batch of image folders by applying masks derived from a baseline folder.
    
    Parameters:
    - input_parent_folder (str): Path to folders containing input images.
    - output_parent_folder (str): Path where filtered images will be saved.
    - baseline_folder (str): Path to reference masks (e.g., baseline predictions).
    - index_range (range): Index range of image filenames to process (e.g., range(26, 50)).
    - threshold_ratio (float): Threshold ratio for binary mask creation (default: 0.71).
    """
    for folder in os.listdir(input_parent_folder):
        if folder.startswith('.'):
            continue  # Skip hidden/system files like .DS_Store

        input_folder = os.path.join(input_parent_folder, folder)
        output_folder = os.path.join(output_parent_folder, folder)

        if os.path.exists(output_folder):
            print(f"Skipping '{folder}': output already exists.")
            continue

        os.makedirs(output_folder)

        # Load input images
        images = []
        for i in index_range:
            image_path = os.path.join(input_folder, f"{i}.png")
            image = Image.open(image_path).convert("L")  # Grayscale
            images.append(image)

        # Load and process baseline masks
        masks_ref = []
        for i in index_range:
            mask_path = os.path.join(baseline_folder, f"{i}.png")
            mask = Image.open(mask_path).convert("L")
            mask_array = np.array(mask)
            
            # Create binary mask based on threshold
            threshold_value = threshold_ratio * 255
            binary_mask = np.where(mask_array < threshold_value, 255, 0).astype(np.uint8)
            dilated_mask = binary_dilation(binary_mask)
            dilated_mask = binary_dilation(dilated_mask)
            
            labeled_mask, num_features = label(dilated_mask)
            if num_features == 0:
                largest_component_mask = binary_mask
            else:
                component_sizes = [np.sum(labeled_mask == idx) for idx in range(1, num_features + 1)]
                largest_component_label = np.argmax(component_sizes) + 1
                largest_component_mask = np.where(labeled_mask == largest_component_label, 255, 0).astype(np.uint8)
            
            convolved_mask = gaussian_filter(largest_component_mask, sigma=1)
            masks_ref.append(convolved_mask)

        # Apply masks and save filtered images
        for idx, (image, mask) in enumerate(zip(images, masks_ref)):
            img_array = np.array(image)
            filtered_image = 255 - mask + img_array * (mask / 255)
            filtered_image = Image.fromarray(filtered_image.astype(np.uint8))
            filename = f"{idx + index_range.start}.png"
            filtered_image_path = os.path.join(output_folder, filename)
            filtered_image.save(filtered_image_path)


def flag_it(
    input_parent_folder, output_parent_folder, baseline_folder,
    index_range=None,
    default_threshold_ratio=None, sup_threshold_ratio=None,
    orange_threshold=None, red_threshold=None
):
    """
    Flags differences between reference masks and input images by applying visual borders and generating diagnostics.

    Parameters:
    - input_parent_folder (str): Folder containing input folders with grayscale images.
    - output_parent_folder (str): Folder where output images (with borders and visualizations) are saved.
    - baseline_folder (str): Folder containing baseline reference masks.
    - index_range (range): Range of image indices to process (default: 26-49).
    - default_threshold_ratio (float): Threshold for binarizing reference masks.
    - sup_threshold_ratio (float): Threshold for binarizing input images for "sup" masks.
    - orange_threshold (tuple): (min, max) range of difference to assign orange border.
    - red_threshold (float): Minimum difference to assign red border.

    Returns:
    - pd.DataFrame: A DataFrame containing folder names, image indices, and difference scores.
    """

    default_threshold = default_threshold_ratio * 255
    sup_threshold = sup_threshold_ratio * 255
    difference_data = []

    for folder in os.listdir(input_parent_folder):
        if folder.startswith('.'):
            continue

        input_folder = os.path.join(input_parent_folder, folder)
        output_folder = os.path.join(output_parent_folder, folder)
        os.makedirs(output_folder, exist_ok=True)

        # Load images
        images = [Image.open(os.path.join(input_folder, f"{i}.png")).convert("L")
                  for i in index_range]

        # Process each image and corresponding baseline mask
        for idx, i in enumerate(index_range):
            mask_path = os.path.join(baseline_folder, f"{i}.png")
            mask = Image.open(mask_path)
            mask_array = np.array(mask)

            # Reference mask processing
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

            # Sup mask processing
            img_array = np.array(images[idx])
            binary_mask_sup = np.where(img_array < sup_threshold, 255, 0).astype(np.uint8)
            labeled_sup, n_sup = label(binary_mask_sup)
            if n_sup > 0:
                largest_label_sup = np.argmax([np.sum(labeled_sup == l) for l in range(1, n_sup + 1)]) + 1
                largest_mask_sup = (labeled_sup == largest_label_sup).astype(np.uint8) * 255
            else:
                largest_mask_sup = binary_mask_sup

            # Filter image using reference mask
            filtered_ref = 255 - convolved_mask_ref + img_array * (convolved_mask_ref / 255)
            filtered_ref = np.clip(filtered_ref, 0, 255).astype(np.uint8)
            filtered_ref_img = Image.fromarray(filtered_ref).convert("RGB")

            # Calculate difference score
            difference = (np.sum(largest_mask_sup * (255. - convolved_mask_ref)) / 255.) / 255

            # Set border color based on difference thresholds
            if orange_threshold[0] <= difference <= orange_threshold[1]:
                border_color = 'orange'
            elif difference > red_threshold:
                border_color = 'red'
            else:
                border_color = 'white'

            # Add border
            filtered_ref_with_border = ImageOps.expand(filtered_ref_img, border=2, fill=border_color)
            save_path = os.path.join(output_folder, f"{i}.png")
            filtered_ref_with_border.save(save_path)

            # Diagnostic difference map
            difference_array = (largest_mask_sup * (255. - convolved_mask_ref)) / 255.
            difference_image = Image.fromarray(difference_array.astype(np.uint8))

            # Save visual diagnostics
            fig, axes = plt.subplots(1, 5, figsize=(12, 4))
            axes[0].imshow(img_array, cmap='gray')
            axes[0].set_title(f'Original ({folder})')
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

            result_fig_path = os.path.join(output_folder, f"result_{i}.png")
            plt.savefig(result_fig_path, bbox_inches='tight')
            plt.close(fig)

            difference_data.append({'Folder': folder, 'Image': i, 'Difference': difference})

    return pd.DataFrame(difference_data)