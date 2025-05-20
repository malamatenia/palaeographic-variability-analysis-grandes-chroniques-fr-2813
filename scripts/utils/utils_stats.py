import json
import os
import re
import unicodedata
import pandas as pd
import numpy as np
import statistics
import rootutils
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from IPython.display import display
from matplotlib.lines import Line2D

from PIL import Image
from collections import defaultdict

#LateX font style 
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

plt.rcParams['text.usetex'] = False



def filter_dataset(input_file: Path, output_no_latin: Path, output_full_lines: Path):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotation_no_latin = {
        k: v for k, v in data.items()
        if not k.startswith("btv1b84472995_f987")
    }

    annotation_full_lines = {
        k: v for k, v in annotation_no_latin.items()
        if v.get("line") not in ["HalfLine", "RubricLine"]
    }

    for path, content in [
        (output_no_latin, annotation_no_latin),
        (output_full_lines, annotation_full_lines)
    ]:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)

    return annotation_no_latin, annotation_full_lines

def compute_macro_statistics(data):
    """
    Compute special character, word, and line frequencies for each script and return them in a DataFrame.
    """
    # Regex to match special characters
    special_char_regex = re.compile(r'[^a-zA-Z ΑΩ¶.*:]')

    # Dictionaries to store aggregated character frequencies, word data, and line counts for each script
    char_frequencies_by_script = {}
    word_data_by_script = {}
    line_counts_by_script = {}
    char_counts_per_line_by_script = {}

    # Initialize storage for the script if not already present
    for _, doc_data in data.items():
        script = doc_data.get('gp', 'Unknown')  # Script category
        label = doc_data.get('label', '')  # Text label

        # Normalize the label to NFC (Normalization Form C)
        normalized_label = unicodedata.normalize('NFD', label)
        normalized_label = normalized_label.replace('\'', '')
        normalized_label = normalized_label.replace('#', '').replace('j', 'i').replace('v', 'u')

        # Initialize storage for the script if not already present
        if script not in char_frequencies_by_script:
            char_frequencies_by_script[script] = []
            word_data_by_script[script] = {'total_words': 0, 'abbreviated_words': 0}
            line_counts_by_script[script] = 0
            char_counts_per_line_by_script[script] = 0

        # Increment the line count for this script
        line_counts_by_script[script] += 1

        # Split the text into words
        words = normalized_label.split()
        word_data_by_script[script]['total_words'] += len(words)

        # Check for abbreviated words
        for word in words:
            if any(special_char_regex.search(char) for char in word):
                word_data_by_script[script]['abbreviated_words'] += 1

        # Append the characters to the corresponding script
        char_frequencies_by_script[script].extend(list(normalized_label))

        # Count the number of characters (excluding spaces) for this line
        char_count_for_this_line = len([char for char in normalized_label if char != ' '])
        char_counts_per_line_by_script[script] += char_count_for_this_line

    # Prepare results to store in DataFrame
    results = []

    # Process the gathered data and compute statistics
    for script, characters in char_frequencies_by_script.items():
        # Remove spaces for special character frequency computation
        characters_no_space = [char for char in characters if char != ' ']

        # Count character occurrences
        char_counts_no_space = pd.Series(characters_no_space).value_counts()
        total_chars = char_counts_no_space.sum()

        # Apply regex to extract special characters
        special_chars = char_counts_no_space[char_counts_no_space.index.map(lambda x: bool(special_char_regex.match(str(x))))]
        special_char_count = special_chars.sum()
        special_char_percentage = round((special_char_count / total_chars) * 100, 1) if total_chars > 0 else 0

        # Word-level statistics
        total_words = word_data_by_script[script]['total_words']
        abbreviated_words = word_data_by_script[script]['abbreviated_words']
        abbreviated_word_percentage = round((abbreviated_words / total_words) * 100, 1) if total_words > 0 else 0

        # Line count for the script
        total_lines = line_counts_by_script[script]

        # Store the results
        results.append({
            "Graphic Profile": script,
            "Total Lines": total_lines,
            "Total Characters": total_chars,
            "Special Characters Count": special_char_count,
            "Special Characters Percentage": special_char_percentage,
            "Graphic Units: Total": total_words,
            "Abbreviated Graphic Units": abbreviated_words,
            "Abbreviated Graphic Units: Percentage": abbreviated_word_percentage,
        })

    # Convert the results list into a DataFrame
    results_df = pd.DataFrame(results)

    # Sort the DataFrame by 'Graphic Profile' to ensure GP1, GP2, GP3, GP4 are in order
    results_df['Graphic Profile'] = pd.Categorical(results_df['Graphic Profile'], categories=['GP1', 'GP2', 'GP3', 'GP4'], ordered=True)
    results_df = results_df.sort_values('Graphic Profile')

    return results_df

def compute_abbreviative_mark_frequencies(data):
    special_char_regex = re.compile(r'[^a-zA-Z ΑΩ¶.*:]') #comment this
    char_frequencies_by_gp = {}
    total_special_chars_by_gp = {}
    
    for _, doc_data in data.items():
        gp = doc_data.get('gp', 'Unknown')
        label = doc_data.get('label', '')
        normalized_label = unicodedata.normalize('NFD', label).replace("'", "").replace("#", "").replace("j", "i").replace("v", "u")
        
        if gp not in char_frequencies_by_gp:
            char_frequencies_by_gp[gp] = []
            total_special_chars_by_gp[gp] = 0
        
        chars = [char for char in normalized_label if char != ' ']
        char_frequencies_by_gp[gp].extend(chars)
    
    df_data = {}
    all_chars = set()
    
    for gp, chars in char_frequencies_by_gp.items():
        char_counts = pd.Series(chars).value_counts()
        special_chars = char_counts[char_counts.index.map(lambda x: bool(special_char_regex.match(str(x))))]
        total_special_chars_by_gp[gp] = special_chars.sum()
        
        df_data[gp] = {char: (count, (count / total_special_chars_by_gp[gp]) * 100 if total_special_chars_by_gp[gp] > 0 else 0) 
                       for char, count in special_chars.items()}
        all_chars.update(special_chars.index)
    
    # Compute the average percentage across all GPs for sorting
    char_avg_percentage = {
        char: np.mean([
            df_data.get(gp, {}).get(char, (0, 0))[1]  # Get the percentage for each GP
            for gp in df_data.keys()
        ])
        for char in all_chars
    }

    # Sort characters by their average percentage across all groups
    all_chars = sorted(char_avg_percentage.keys(), key=lambda char: char_avg_percentage[char], reverse=True)

    df = pd.DataFrame(index=all_chars)
    
    for gp in sorted(df_data.keys()):
        df[f'{gp} Count (%)'] = df.index.map(lambda char: df_data[gp].get(char, (0, 0)))
    
    df = df.map(lambda x: f'{x[0]} ({x[1]:.2f}%)' if isinstance(x, tuple) else x)
    
    df['Avg %'] = df.index.map(lambda char: char_avg_percentage.get(char, 0))
    top_15_chars = df.nlargest(15, 'Avg %')
    top_15_chars = top_15_chars.drop(columns=['Avg %'])
    
    return top_15_chars

def plot_abbreviative_mark_histogram(df,
    stats_dir, 
    junicode_font,
    legend_labels,
    gp_colors,
    log_scale=False):

    legend_colors = [gp_colors[label] for label in legend_labels]

    plt.figure(figsize=(14, 8))
    
    df_numeric = df.map(lambda x: float(x.split('(')[1][:-2]) if isinstance(x, str) and '(' in x else 0)

    if log_scale:
        df_numeric = df_numeric.map(lambda x: np.log(x + 1) if x > 0 else 0)

    ax = df_numeric.plot(kind='bar', width=0.8, color=legend_colors, figsize=(14, 8), position=0)

    if log_scale:
        ax.set_yscale("log")

    if plt.rcParams.get('text.usetex', False):
        ax.set_ylabel(r'$\textbf{Percentage}$', fontsize=19)
    else:
        ax.set_ylabel('Percentage', fontsize=19)

    xticks_pos = np.arange(len(df_numeric.index))
    ax.set_xticks(xticks_pos + 0.25)
    ax.set_xticklabels(df_numeric.index, rotation=0, fontproperties=junicode_font, fontsize=21, fontweight='bold')

    for label in ax.get_xticklabels():
        label.set_bbox(dict(facecolor='white', edgecolor='grey', boxstyle='square,pad=0.25'))

    legend_patches = [mpatches.Patch(color=gp_colors[label], label=label) for label in legend_labels]
    
    if plt.rcParams.get('text.usetex', False):
        ax.legend(title=r'$\textbf{Graphic Profiles}$', handles=legend_patches, loc='upper right', title_fontsize=20, fontsize=17)
    else:
        ax.legend(title="Graphic Profiles", handles=legend_patches, loc='upper right', title_fontsize=20, fontsize=17)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    histogram_path = stats_dir / "abbreviative_mark_frequency_per_GP.png"
    plt.savefig(histogram_path)
    plt.show()

    return df_numeric

def compute_sequence_abbreviation_frequency(data,
    reverse_abbreviation_map,
    legend_labels,
    stats_dir):
    """
    Compute abbreviation percentages and detailed counts per script.
    """
    script_abbreviation_data = {
        gp: {abbr: {"abbr": 0, "unabbr": 0} for abbr in reverse_abbreviation_map}
        for gp in legend_labels
    }
    scribe_labels = {}

    for _, doc_data in data.items():
        label = doc_data.get('label', '').lower()
        script = doc_data.get('gp', 'Unknown')

        if script not in scribe_labels:
            scribe_labels[script] = ""
        scribe_labels[script] += label

    for script, combined_label in scribe_labels.items():
        normalized_label = unicodedata.normalize('NFD', combined_label)

        for abbr, full_list in reverse_abbreviation_map.items():
            abbr_pattern = unicodedata.normalize('NFD', abbr)
            abbr_count = len(re.findall(re.escape(abbr_pattern), normalized_label, flags=re.UNICODE))

            full_count = sum(
                len(re.findall(re.escape(unicodedata.normalize('NFD', full)), normalized_label, flags=re.UNICODE))
                for full in full_list
            )

            script_abbreviation_data[script][abbr]["abbr"] += abbr_count
            script_abbreviation_data[script][abbr]["unabbr"] += full_count

    abbreviation_data = []
    for abbr, full_list in reverse_abbreviation_map.items():
        row = {
            "Abbreviated Sequence": abbr,
            "Unabbreviated Sequences": ", ".join(full_list)
        }

        for script in legend_labels:
            counts = script_abbreviation_data.get(script, {}).get(abbr, {"abbr": 0, "unabbr": 0})
            abbr_count = counts["abbr"]
            unabbr_count = counts["unabbr"]
            total = abbr_count + unabbr_count
            abbr_percentage = (abbr_count / total * 100) if total > 0 else 0

            row[f"{script} Unabbr."] = unabbr_count
            row[f"{script} Abbr."] = abbr_count
            row[f"{script} Total"] = total
            row[f"{script} % of abbr."] = round(abbr_percentage, 1)

        abbreviation_data.append(row)

    df = pd.DataFrame(abbreviation_data)
    percentage_columns = [f"{script} % of abbr." for script in legend_labels]
    df["Average %"] = df[percentage_columns].mean(axis=1)
    df = df.sort_values(by="Average %", ascending=False)

    # Save CSV
    csv_path = stats_dir / "syllabic_sequence_abbreviation_per_GP.csv"
    df.to_csv(csv_path, index=False)

    display(df)
    return df


def plot_sequence_abbreviation_histogram(
    df,
    legend_labels,
    gp_colors,
    junicode_font,
    stats_dir,
    abbreviation_subset=None):

    """
    Plot a grouped bar histogram for abbreviation percentage per script.
    """
    if abbreviation_subset is not None:
        df = df[df["Abbreviated Sequence"].isin(abbreviation_subset)]

    df = df.set_index("Abbreviated Sequence")
    df_numeric = df[[f"{script} % of abbr." for script in legend_labels]]

    legend_colors = [gp_colors[label] for label in legend_labels]

    fig, ax = plt.subplots(figsize=(14, 8))
    df_numeric.plot(kind='bar', width=0.8, color=legend_colors, ax=ax)

    if plt.rcParams.get('text.usetex', False):
        ax.set_ylabel(r'\textbf{Abbreviation Usage (\%)}', fontsize=19)
    else:
        ax.set_ylabel('Abbreviation Usage (%)', fontsize=19)

    ax.set_xlabel(None)
    ax.set_xticklabels(df.index, rotation=0)

    for label in ax.get_xticklabels():
        label.set_fontproperties(junicode_font)
        label.set_fontsize(21)
        label.set_fontweight('bold')
        label.set_usetex(False)
        label.set_bbox(dict(facecolor='white', edgecolor='grey', boxstyle='square,pad=0.25'))

    legend_patches = [mpatches.Patch(color=gp_colors[label], label=label) for label in legend_labels]
    ax.legend(title="Graphic Profiles", handles=legend_patches, loc='upper right', title_fontsize=20, fontsize=17)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    histogram_path = stats_dir / "syllabic_sequence_abbreviation_per_GP.png"
    plt.savefig(histogram_path)
    plt.show()

def compute_relative_position_histogram(
    data,
    font_path,
    output_path,
    count_threshold=None,
    legend_labels=None,
    gp_colors=None
):
    """
    Compute and plot a histogram showing the relative positions of special characters per script.
    Saves both the DataFrame (CSV) and the figure (PNG).
    """

    non_alpha_regex = re.compile(r'[^a-zA-Z ΑΩ̶¶.*:]')
    num_bins = 10
    aggregated_bins = {}

    # Analyze each document
    for _, doc in data.items():
        script = doc.get("gp", "Unknown")
        label = doc.get("label", "")
        if not label:
            continue

        normalized = unicodedata.normalize("NFD", label).replace("#", "").replace("j", "i").replace("v", "u")
        special_char_positions = [i for i, c in enumerate(normalized) if non_alpha_regex.match(c)]

        if not special_char_positions:
            continue

        text_len = len(normalized)
        bin_edges = [int(i * text_len / num_bins) for i in range(num_bins)] + [text_len]

        if script not in aggregated_bins:
            aggregated_bins[script] = [0] * num_bins

        for pos in special_char_positions:
            for i in range(num_bins):
                if bin_edges[i] <= pos < bin_edges[i + 1]:
                    aggregated_bins[script][i] += 1
                    break

    # Create DataFrame and normalize
    bin_labels = [f"Bin {i+1}" for i in range(num_bins)]
    df = pd.DataFrame.from_dict(aggregated_bins, orient="index", columns=bin_labels)
    df = df.div(df.sum(axis=1), axis=0) * 100  # Convert to percentages

    # Save CSV
    csv_path = os.path.join(output_path, "relative_position_special_characters_per_GP.csv")
    df.to_csv(csv_path)

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(num_bins)

    if legend_labels is None:
        legend_labels = df.index.tolist()
    if gp_colors is None:
        gp_colors = {label: f"C{i}" for i, label in enumerate(legend_labels)}

    bar_width = 0.8 / len(legend_labels)

    for idx, script in enumerate(legend_labels):
        if script not in df.index:
            continue
        color = gp_colors.get(script, f"C{idx}")
        ax.bar(
            x + (idx - (len(legend_labels) - 1) / 2) * bar_width,
            df.loc[script],
            width=bar_width,
            label=script,
            edgecolor='black',
            color=color
        )

    title_text = (
        r'$\textbf{Relative\ Position\ of\ Special\ Characters}$'
        if plt.rcParams.get('text.usetex', False)
        else 'Relative Position of Special Characters'
    )

    ax.text(
        0.5, 0.95,
        title_text,
        fontsize=25, ha='center', va='top', transform=ax.transAxes
    )

    ylabel_text = (
        r'$\textbf{Percentage}$'
        if plt.rcParams.get('text.usetex', False)
        else 'Percentage'
    )

    ax.set_ylabel(ylabel_text, fontsize=19)

    # X-axis bins
    font = FontProperties(fname=font_path)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=19, fontproperties=font)

    ax.legend(title="Graphic Profiles", title_fontsize=20, fontsize=17)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    fig_path = os.path.join(output_path, "relative_position_special_characters_per_GP.png")
    plt.savefig(fig_path)
    plt.show()
    plt.close(fig)
    

def natural_sort_key(text):
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', text)]


def create_metadata_mapping(annotation_json_path):
    with open(annotation_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    folio_mapping = {}
    graphic_profile_mapping = {}

    for filename_with_ext, metadata in data.items():
        filename = filename_with_ext.replace(".png", "")
        parts = filename.split('_')
        if len(parts) < 2:
            continue

        folder_name = '_'.join(parts[:2])
        folio_mapping[folder_name] = metadata.get('folio')
        graphic_profile_mapping[folder_name] = metadata.get('gp', 'Unknown')

    return folio_mapping, graphic_profile_mapping


def calculate_line_statistics(images_dir, base_images_dir, annotation_json_path, output_path, dpi):
    pixel_to_mm = 25.4 / dpi

    with open(annotation_json_path, 'r') as f:
        json_metadata = json.load(f)

    folio_mapping, graphic_profile_mapping = create_metadata_mapping(annotation_json_path)

    avg_lengths_per_subfolder = {}

    graphic_profile_lengths = {}
    graphic_profile_std_dev = {}
    graphic_profile_total_chars = {}
    graphic_profile_total_lines = {}

    subfolders = sorted(
        [folder for folder in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, folder))],
        key=natural_sort_key
    )

    for subfolder in subfolders:
        lengths = []
        total_characters = 0
        total_lines = 0

        folio = folio_mapping.get(subfolder, 'Unknown')
        graphic_profile = graphic_profile_mapping.get(subfolder, 'Unknown')

        base_subfolder = re.sub(r'([a-z])$', '', subfolder)
        base_image_path = os.path.join(base_images_dir, f"{base_subfolder}.jpg")

        if not os.path.isfile(base_image_path):
            print(f"Base image not found for subfolder '{subfolder}': {base_image_path}")
            continue

        try:
            with Image.open(base_image_path) as base_img:
                base_width, _ = base_img.size
        except Exception as e:
            print(f"Error processing base image {base_image_path}: {e}")
            continue

        for filename_with_ext, metadata in json_metadata.items():
            filename = filename_with_ext.replace(".png", "")
            parts = filename.split('_')
            if len(parts) < 2:
                continue
            folder_from_filename = '_'.join(parts[:2])

            if folder_from_filename != subfolder:
                continue

            label = metadata.get('label', '')
            if not label:
                print(f"Skipping file {filename_with_ext} due to missing label.")
                continue

            line_length = len(label)
            total_characters += line_length
            total_lines += 1

            file_path = os.path.join(images_dir, subfolder, filename_with_ext)
            if not os.path.isfile(file_path):
                print(f"Image file {file_path} not found for corresponding JSON entry.")
                continue

            try:
                with Image.open(file_path) as img:
                    width, _ = img.size
                    length_mm = (width / base_width) * (base_width * pixel_to_mm)
                    lengths.append(length_mm)
            except Exception as e:
                print(f"Error processing image {file_path}: {e}")

        avg_length_mm = sum(lengths) / len(lengths) if lengths else 0
        avg_char_per_line = total_characters / total_lines if total_lines else 0
        standard_deviation = statistics.stdev(lengths) if len(lengths) > 1 else 0

        avg_lengths_per_subfolder[subfolder] = {
            "Folio": folio,
            "Graphic Profile": graphic_profile,
            "Average Line Length (mm)": avg_length_mm,
            "Average Characters per Line": avg_char_per_line,
            "Standard Deviation (mm)": standard_deviation,
            "Total Lines": total_lines,
        }

        if graphic_profile not in graphic_profile_lengths:
            graphic_profile_lengths[graphic_profile] = []
            graphic_profile_std_dev[graphic_profile] = []
            graphic_profile_total_chars[graphic_profile] = 0
            graphic_profile_total_lines[graphic_profile] = 0

        graphic_profile_lengths[graphic_profile].extend(lengths)
        graphic_profile_std_dev[graphic_profile].append(standard_deviation)
        graphic_profile_total_chars[graphic_profile] += total_characters
        graphic_profile_total_lines[graphic_profile] += total_lines

    avg_stats_per_gp = []
    for gp in graphic_profile_lengths:
        lengths = graphic_profile_lengths[gp]
        total_chars = graphic_profile_total_chars[gp]
        total_lines = graphic_profile_total_lines[gp]
        stds = graphic_profile_std_dev.get(gp, [])

        avg_length = round(sum(lengths) / len(lengths), 1) if lengths else 0
        avg_char = round(total_chars / total_lines, 1) if total_lines > 0 else 0
        avg_std_dev = round(sum(stds) / len(stds), 1) if stds else 0

        avg_stats_per_gp.append({
            "Graphic Profile": gp,
            "Average Line Length (mm)": avg_length,
            "Average Characters per Line": avg_char,
            "Average Standard Deviation (mm)": avg_std_dev,
            "Total Characters": total_chars,
            "Total Lines": total_lines
        })

    df_gp_stats = pd.DataFrame(avg_stats_per_gp)
    df_avg_lengths_per_subfolder = pd.DataFrame.from_dict(avg_lengths_per_subfolder, orient='index').fillna(0)

    subfolder_csv_path = os.path.join(output_path, "average_lengths_per_folio.csv")
    df_avg_lengths_per_subfolder.to_csv(subfolder_csv_path, index_label="Subfolder")

    gp_stats_csv_path = os.path.join(output_path, "average_line_statistics_per_GP.csv")
    df_gp_stats.to_csv(gp_stats_csv_path, index=False)
    print(f"Saved 'Average Statistics per Graphic Profile' to '{gp_stats_csv_path}'")
    
    display(df_gp_stats)
    return df_avg_lengths_per_subfolder, df_gp_stats
    


def create_line_scatter_plot(subfolder_df, gp_mapping, folio_mapping, output_path, gp_colors, gp_markers):

    fig, ax = plt.subplots(figsize=(12, 8))

    gps_in_data = set()
    plotted_count = 0
    skipped_count = 0
    skipped_few_lines_count = 0

    for subfolder, row in subfolder_df.iterrows():
        gp = gp_mapping.get(subfolder)
        if gp is None or gp == "Unknown":
            skipped_count += 1
            continue

        total_lines = row.get("Total Lines", None)
        if total_lines is not None and total_lines < 15:
            skipped_few_lines_count += 1
            continue

        folio = folio_mapping.get(subfolder, "Unknown")
        gps_in_data.add(gp)
        plotted_count += 1

        color = gp_colors.get(gp, 'gray')
        marker = gp_markers.get(gp, 'o')

        ax.scatter(
            row["Average Line Length (mm)"],
            row["Average Characters per Line"],
            color=color,
            marker=marker,
            s=90,
            alpha=0.9,
            edgecolors='black'
        )

        ax.annotate(
            folio,
            (row["Average Line Length (mm)"], row["Average Characters per Line"]),
            fontsize=10,
            alpha=0.7,
            xytext=(6, 6),
            textcoords='offset points',
        )

    print(f"Plotted {plotted_count} subfolders.")
    print(f"Skipped {skipped_count} subfolders without GP mapping.")
    print(f"Skipped {skipped_few_lines_count} subfolders with less than 15 lines.")
    if plt.rcParams.get('text.usetex', False):
        ax.set_xlabel(r'$\textbf{Average Line Length (mm)}$', fontsize=18)
        ax.set_ylabel(r'$\textbf{Average Characters Per Line}$', fontsize=18)
    else:
        ax.set_xlabel('Average Line Length (mm)', fontsize=18)
        ax.set_ylabel('Average Characters Per Line', fontsize=18)

    legend_elements = [
        Line2D([0], [0], marker=gp_markers[gp], color='w',
               markerfacecolor=gp_colors[gp], markeredgecolor='black',
               markersize=12, label=gp)
        for gp in sorted(gps_in_data) if gp in gp_colors
    ]
    ax.legend(title="Graphic Profiles", handles=legend_elements, loc='upper left', title_fontsize=16, fontsize=14)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    plt.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Scatter plot saved to {output_path}")
    plt.show()