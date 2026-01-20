
from .visualisations import plot_box, show_reading_order, show_reading_order_df

import numpy as np
from skimage.filters import threshold_otsu
import pandas as pd


def find_columns(bin_image, threshold=None, min_col_area=None, pad=True):
    """
    Find the columns in a binary image.
    Args:
        bin_image, the image
        threshold. see the report
        min_col_area. minimun area of a column
        pad: set to true to handle columns touching image borders

    Returns:
        left_cols: pixel coordinates of the left side of each column
        right_cols: pixel coordinates of the right side of each column
    """

    proj = bin_image.sum(0)

    if pad:
        proj = np.concatenate([[0], proj, [0]])

    if threshold is None:
        threshold = threshold_otsu(proj)

    cond = (proj < threshold).astype(int)
    diff = cond[1:]-cond[:-1]
    left_cols = np.where(diff==-1)[0]
    right_cols = np.where(diff==1)[0]


    if min_col_area is not None:
        has_sufficient_area = np.zeros(left_cols.shape, dtype=bool)
        for col_idx, (l, r) in enumerate(zip(left_cols, right_cols)):
            area = proj[l:r].sum()
            has_sufficient_area[col_idx] = area > min_col_area
        left_cols = left_cols[has_sufficient_area]
        right_cols = right_cols[has_sufficient_area]

    if pad:
        left_cols = left_cols - 1
        right_cols = right_cols - 1

    return left_cols, right_cols


def find_columns_by_label(label_image, threshold=None, min_col_area=None, pad=True):
    """
    Find one column per label class in a labeled image.
    
    Args:
        label_image: labeled image where each connected component has a unique integer label
=        pad: unused (kept for signature compatibility)
    
    Returns:
        left_cols: pixel coordinates of the left side of each column
        right_cols: pixel coordinates of the right side of each column
    """
    
    labels = np.unique(label_image)
    labels = labels[labels != 0]  # Remove background
    
    if len(labels) == 0:
        return np.array([]), np.array([])
    
    # Create a mask of which labels appear in each column
    # Shape: (num_labels, num_cols)
    label_presence = label_image[None, :, :] == labels[:, None, None]
    cols_per_label = label_presence.any(axis=1)  # (num_labels, num_cols)
    
    # Find first and last column for each label
    col_indices = np.arange(label_image.shape[1])
    left_cols = np.array([col_indices[cols].min() for cols in cols_per_label])
    right_cols = np.array([col_indices[cols].max() + 1 for cols in cols_per_label])
    
    # Sort by left column position (document order)
    sort_idx = np.argsort(left_cols)
    
    return left_cols[sort_idx], right_cols[sort_idx]

def split_to_rectangles(labels, min_col_area, threshold_cols=1, canvas=None, threshold_label_area=10):
    """ Returns a dataframe containing every col/row of the document and the labels in it """
    
    
    bin_image = labels != 0

    rectangles = []
    left_cols, right_cols = find_columns(bin_image, threshold = threshold_cols, min_col_area=min_col_area)

    # For every column
    for col_idx, (left, right) in enumerate(zip(left_cols, right_cols)):
        column_slice = bin_image[:, left:right]
        lbls_slice = labels[:, left:right]

        # For every row in the column
        # top_rows, bottom_rows = find_columns_by_label(lbls_slice.T)
        
        top_rows, bottom_rows = find_columns(column_slice.T)

        for row_idx, (top, bottom) in enumerate(zip(top_rows, bottom_rows)):

            crop = column_slice[top:bottom]
            lbl_slice = labels[top:bottom, left:right]

            # For every subcolumn
            subcol_lefts, subcol_rights = find_columns(crop, threshold=1)
            crop_labels = []

            for subcol_left, subcol_right in zip(subcol_lefts, subcol_rights):
                # Add the dominant label of the subcolumn to the crop
                unique_labels, counts = np.unique(lbl_slice[:, subcol_left:subcol_right], return_counts=True)
                dominant_label = unique_labels[-1]
                crop_labels.append(dominant_label)
            
            # Add the row to the dataframe
            rectangle = {
                'col_idx': col_idx,
                'row_idx': row_idx,
                'bbox': (left, top, right, bottom),
                'labels': crop_labels
            }
            rectangles.append(rectangle)

            # Optionally, plot the box on the canvas
            if canvas is not None:
                plot_box(canvas, rectangle)
            
    rectangles = pd.DataFrame(rectangles)
    return rectangles


def break_into_subcols(column_rectangles: pd.DataFrame):
    """ Takes a single column and returns a list of subcolumns of same size """
    column_rectangles = column_rectangles.set_index('row_idx', drop=False) # to iterate safely

    subcol_start = 0
    dataframes = []
    n_cols = [len(column_rectangles['labels'].iloc[0])]  # Use iloc for first row
    max_row = column_rectangles.index.max()

    for row_idx, row in column_rectangles.iterrows():
        n = len(row['labels'])
        if n != n_cols[-1]:
            # Add previous subcolumn (up to but not including current row)
            dataframes.append(column_rectangles.loc[subcol_start:row_idx-1])
            n_cols.append(n)
            subcol_start = row_idx
    
    # Don't forget the last subcolumn
    dataframes.append(column_rectangles.loc[subcol_start:max_row])
    
    return dataframes


def get_reading_order(rectangles):
    """ Returns the labels of the rectangles, in reading order """
    ordered_labels = []

    # For every column
    for col_idx, col_rectangles in rectangles.groupby('col_idx'):
        subcolumns = break_into_subcols(col_rectangles)

        # For every subcolumn in the column
        for subcolumn in subcolumns:
            n_elems = len(subcolumn['labels'].iloc[0])
            # For every column of the subcolumn
            for subcol_idx in range(n_elems):
                # For every row of the column of the subcolumn
                for idx, row in subcolumn.iterrows():
                    # Add the label to the list
                    ordered_labels.append(row['labels'][subcol_idx])


    return ordered_labels


class ReadingOrder:
    def __init__(
            self,
            min_col_area
    ):
        self.min_col_area = min_col_area

    def __call__(
            self,
            labels,
            page_dataframe,
            canvas_rectangles = None,
            canvas_reading_order = None
    ):
        rectangles = split_to_rectangles(labels, min_col_area=self.min_col_area, canvas=canvas_rectangles)
        ordered_labels = get_reading_order(rectangles)
        mapping = {value: idx for idx, value in enumerate(ordered_labels)}
        page_dataframe['reading_order'] = pd.to_numeric(page_dataframe['label'].map(mapping)).astype("Int64")

        if canvas_reading_order is not None:
            show_reading_order_df(canvas_reading_order, page_dataframe)

        if canvas_rectangles is not None:
            fig = show_reading_order(canvas_rectangles, ordered_labels, rectangles)
            return fig
        return None
