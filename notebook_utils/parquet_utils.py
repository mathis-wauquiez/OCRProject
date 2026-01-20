import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import gc


class ColumnSaver:
    """Base class for column savers"""
    
    @staticmethod
    def can_handle(col_data, first_val):
        raise NotImplementedError
    
    @staticmethod
    def save(col_data, filepath):
        raise NotImplementedError
    
    @staticmethod
    def load(filepath):
        raise NotImplementedError
    
    @staticmethod
    def metadata(col_data, first_val):
        return {}


class NumericSaver(ColumnSaver):
    """Handles numeric columns"""
    
    @staticmethod
    def can_handle(col_data, first_val):
        return col_data.dtype in ['int64', 'int32', 'float64', 'float32', 'bool']
    
    @staticmethod
    def save(col_data, filepath):
        pd.DataFrame(col_data).to_parquet(filepath)
    
    @staticmethod
    def load(filepath):
        return pd.read_parquet(filepath).iloc[:, 0].values
    
    @staticmethod
    def metadata(col_data, first_val):
        return {'type': 'numeric', 'dtype': str(col_data.dtype)}


class StringSaver(ColumnSaver):
    """Handles string columns"""
    
    @staticmethod
    def can_handle(col_data, first_val):
        return col_data.dtype == 'object' and isinstance(first_val, str)
    
    @staticmethod
    def save(col_data, filepath):
        pd.DataFrame(col_data).to_parquet(filepath)
    
    @staticmethod
    def load(filepath):
        return pd.read_parquet(filepath).iloc[:, 0].values
    
    @staticmethod
    def metadata(col_data, first_val):
        return {'type': 'string'}


class NumpyArraySaver(ColumnSaver):
    """Handles numpy array columns"""
    
    @staticmethod
    def can_handle(col_data, first_val):
        return isinstance(first_val, np.ndarray)
    
    @staticmethod
    def save(col_data, filepath):
        arrays = {str(i): arr for i, arr in enumerate(col_data.values)}
        np.savez_compressed(filepath, **arrays)
    
    @staticmethod
    def load(filepath):
        data = np.load(filepath)
        arrays = [data[str(i)] for i in range(len(data.files))]
        data.close()
        return arrays
    
    @staticmethod
    def metadata(col_data, first_val):
        return {
            'type': 'numpy_array',
            'shape': first_val.shape,
            'dtype': str(first_val.dtype)
        }


class ObjectSaver(ColumnSaver):
    """Handles custom objects (SVG, etc.)"""
    
    @staticmethod
    def can_handle(col_data, first_val):
        return True  # Fallback for everything else
    
    @staticmethod
    def save(col_data, filepath, chunk_size=1000):
        data_list = col_data.tolist()
        with open(filepath, 'wb') as f:
            pickle.dump(len(data_list), f, protocol=4)
            for i in range(0, len(data_list), chunk_size):
                chunk = data_list[i:i+chunk_size]
                pickle.dump(chunk, f, protocol=4)
    
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            length = pickle.load(f)
            all_data = []
            while len(all_data) < length:
                chunk = pickle.load(f)
                all_data.extend(chunk)
            return all_data
    
    @staticmethod
    def metadata(col_data, first_val):
        return {'type': 'object', 'class': type(first_val).__name__}


# Registry of savers (order matters - first match wins)
SAVERS = [NumericSaver, StringSaver, NumpyArraySaver, ObjectSaver]


def save_dataframe(df, base_path):
    """Save DataFrame column-by-column"""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'columns': list(df.columns),
        'index': df.index.tolist(),
        'n_rows': len(df)
    }
    
    for col in df.columns:
        print(f"Saving column: {col}")
        col_data = df[col]
        first_val = col_data.iloc[0] if len(col_data) > 0 else None
        
        # Find appropriate saver
        for saver_class in SAVERS:
            if saver_class.can_handle(col_data, first_val):
                # Determine file extension
                ext = '.parquet' if saver_class in [NumericSaver, StringSaver] else \
                      '.npz' if saver_class == NumpyArraySaver else '.pkl'
                
                filepath = base_path / f"{col}{ext}"
                saver_class.save(col_data, filepath)
                metadata[col] = saver_class.metadata(col_data, first_val)
                break
        
        gc.collect()
    
    with open(base_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved to {base_path}")


def load_dataframe(base_path):
    """Load DataFrame from column-by-column storage"""
    base_path = Path(base_path)
    
    with open(base_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    df = pd.DataFrame(index=metadata['index'])
    
    for col in metadata['columns']:
        print(f"Loading column: {col}")
        col_meta = metadata[col]
        col_type = col_meta['type']
        
        # Find appropriate saver by type
        saver_class = {
            'numeric': NumericSaver,
            'string': StringSaver,
            'numpy_array': NumpyArraySaver,
            'object': ObjectSaver
        }[col_type]
        
        # Determine file extension
        ext = '.parquet' if col_type in ['numeric', 'string'] else \
              '.npz' if col_type == 'numpy_array' else '.pkl'
        
        filepath = base_path / f"{col}{ext}"
        df[col] = saver_class.load(filepath)
        
        gc.collect()
    
    print(f"✓ Loaded from {base_path}")
    return df


def load_columns(base_path, columns_to_load=None):
    """Load specific columns from column-by-column storage"""
    base_path = Path(base_path)
    
    with open(base_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # If no columns specified, load all
    if columns_to_load is None:
        columns_to_load = metadata['columns']
    
    # Create empty DataFrame with index
    df = pd.DataFrame(index=metadata['index'])
    
    for col in columns_to_load:
        if col not in metadata['columns']:
            print(f"Warning: Column '{col}' not found in metadata")
            continue
            
        print(f"Loading column: {col}")
        col_meta = metadata[col]
        col_type = col_meta['type']
        
        # Find appropriate saver by type
        saver_class = {
            'numeric': NumericSaver,
            'string': StringSaver,
            'numpy_array': NumpyArraySaver,
            'object': ObjectSaver
        }[col_type]
        
        # Determine file extension
        ext = '.parquet' if col_type in ['numeric', 'string'] else \
              '.npz' if col_type == 'numpy_array' else '.pkl'
        
        filepath = base_path / f"{col}{ext}"
        df[col] = saver_class.load(filepath)
        
        gc.collect()
    
    print(f"✓ Loaded {len(columns_to_load)} columns from {base_path}")
    return df

# Usage
# save_dataframe(patches_df, 'data/processed/book1_columnwise')
# patches_df = load_dataframe('data/processed/book1_columnwise')