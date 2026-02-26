
import cv2
import numpy as np

import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu

def plot_box(canvas, rectangle):
    left, top, right, bottom = rectangle['bbox']
    cv2.rectangle(canvas, (left, top), (right, bottom), (255,255,255), 1)
    text_x = left + 1
    text_y = top + 7
    
    # cv2.putText(canvas, str(len(rectangle['labels'])), (text_x, text_y), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1)


def plot_layout(canvas, labels, min_col_area):
    from .parsing import find_columns
    # Create figure with 2 subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                    gridspec_kw={'height_ratios': [1, 3]})

    bin_image = labels != 0

    proj = bin_image.sum(0)
    otsu = threshold_otsu(proj)
    left_cols, right_cols = find_columns(bin_image, threshold=1, min_col_area=min_col_area)
    left_cols_all, right_cols_all = find_columns(bin_image, threshold=1, min_col_area=None)

    # Calculate areas for all columns to show the threshold
    all_areas = []
    for l, r in zip(left_cols_all, right_cols_all):
        area = proj[l:r].sum()
        all_areas.append(area)

    # Top plot: projection
    ymax = proj.max() * 1.1
    ax1.plot(proj, linewidth=1, label='Projection')

    # Show all columns (including filtered ones) in light colors
    ax1.vlines(left_cols_all, ymin=0, ymax=ymax, colors="lightcoral", 
            linestyles='--', linewidths=1, alpha=0.5, label='Filtered out (left)')
    ax1.vlines(right_cols_all, ymin=0, ymax=ymax, colors="lightgreen", 
            linestyles='--', linewidths=1, alpha=0.5, label='Filtered out (right)')

    # Show kept columns in bright colors
    ax1.vlines(left_cols, ymin=0, ymax=ymax, colors="orange", 
            linestyles='--', linewidths=2, label='Kept (left)')
    ax1.vlines(right_cols, ymin=0, ymax=ymax, colors="green", 
            linestyles='--', linewidths=2, label='Kept (right)')

    # Thresholds
    ax1.hlines(y=[otsu], xmin=0, xmax=len(proj), colors="black", 
            linestyles='--', linewidths=1, label=f'Otsu: {otsu:.1f}', alpha=0.5)
    ax1.hlines(y=[1], xmin=0, xmax=len(proj), colors="red", 
            linestyles='--', linewidths=2, label=f'Manual: {1}')

    # Add min_col_area annotation
    if min_col_area is not None:
        ax1.text(0.02, 0.98, f'Min column area: {min_col_area}\nKept: {len(left_cols)}/{len(left_cols_all)} columns', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)

    ax1.set_xlim(0, len(proj))
    ax1.set_ylabel('Projection Sum')
    ax1.set_xlabel('X Position')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Bottom plot: image
    ax2.imshow(canvas, interpolation='none', aspect='auto')
    ax2.set_xlim(0, canvas.shape[1])
    ax2.set_xlabel('X Position')
    ax2.axis('off')

    # Remove horizontal space between subplots
    plt.subplots_adjust(hspace=0.05)
    plt.tight_layout()
    
    return fig



def show_reading_order(canvas, ordered_labels, rectangles):
    # Draw bounding boxes with reading order numbers
    counter = 0
    for label in ordered_labels:
        # Find the rectangle containing this label
        for _, rect in rectangles.iterrows():
            if label in rect['labels']:
                # Get position within the labels list
                subcol_idx = rect['labels'].index(label)
                
                # Draw the number
                left, top, right, bottom = rect['bbox']
                # text_x = left + (right - left) * (subcol_idx + 0.5) / len(rect['labels'])
                # text_y = top + (bottom - top) // 2

                text_x = left + 1
                text_y = top + 7
                
                cv2.putText(canvas, str(counter), (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1)


                counter += 1
                break

    # Visualize with matplotlib (OO API — safe in threads)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(canvas, interpolation='none', aspect='auto')
    ax.set_title(f'Reading Order ({len(ordered_labels)} elements)')
    ax.axis('off')
    return fig


def show_reading_order_df(canvas, page_patches_df):
    for i, row in page_patches_df.iterrows():
        left, top, width, height = row[['left', 'top', 'width', 'height']]
        cv2.rectangle(canvas, (left, top), (left+width, top+height), (255,255,255), 10)
        import pandas as pd
        if pd.isna(row['reading_order']):
            cv2.rectangle(canvas, (left, top), (left+width, top+height), (0,255,0), 10)
        text_x = left + 1
        text_y = top + 7
        
        cv2.putText(canvas, str(row['reading_order']), (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
