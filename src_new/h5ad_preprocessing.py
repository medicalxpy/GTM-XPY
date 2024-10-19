import scanpy as sc
import scipy.sparse as sp
import argparse
import pandas as pd
import re

def check_if_all_strings(column):
    """
    Check if all elements in the given column are strings.
    """
    if isinstance(column.dtype, pd.CategoricalDtype):
        column = column.astype(str)
    return column.head(10).apply(lambda x: isinstance(x, str)).all()

def remove_version_number(ensemble_id):
    """
    Remove the version number from the ensemble ID.
    """
    return re.sub(r'\.\d+$', '', ensemble_id)

def H5adPreprocessing(file_input_path,
                      cell_ge_expr_threshold=1000,
                      gene_num_threshold=60,
                      normalization=True,
                      log1p=False):
    """
    Preprocess the AnnData object from an h5ad file.

    Parameters:
    - file_input_path: Path to the input h5ad file.
    - cell_ge_expr_threshold: Threshold for cell gene expression.
    - gene_num_threshold: Minimum number of genes per cell.
    - normalization: Whether to perform normalization.
    - log1p: Whether to apply log1p transformation.

    Returns:
    - Preprocessed AnnData object.
    """
    adata = sc.read_h5ad(file_input_path)

    # Filter the cells based on gene count threshold
    sc.pp.filter_cells(adata, min_genes=gene_num_threshold)
    
    # Apply log1p transformation if specified
    if log1p:
        sc.pp.log1p(adata)

    # Perform normalization if specified
    if normalization:
        sc.pp.normalize_total(adata, target_sum=1)  
    
    # Find and update the ensemble_id column
    for index in adata.var.columns:
        if check_if_all_strings(adata.var[index]):
            if adata.var[index].head(10).str.startswith('ENSG').all():
                adata.var[index] = adata.var[index].apply(remove_version_number)
                adata.var_names = adata.var[index]
                break
    
    return adata

def main():
    """
    Main function to parse command-line arguments and preprocess the data.
    """
    parser = argparse.ArgumentParser(description='Preprocess h5ad file.')
    parser.add_argument('--file-input-path', type=str, required=True, help='Path to the input h5ad file.')
    parser.add_argument('--cell-ge-expr-threshold', type=int, default=1000, help='Threshold for cell gene expression.')
    parser.add_argument('--gene-num-threshold', type=int, default=60, help='Threshold for minimum number of genes per cell.')
    parser.add_argument('--normalization', action='store_true', help='Perform normalization.')
    parser.add_argument('--log1p', action='store_true', help='Apply log1p transformation.')

    args = parser.parse_args()

    # Preprocess the data
    adata = H5adPreprocessing(
        file_input_path=args.file_input_path,
        cell_ge_expr_threshold=args.cell_ge_expr_threshold,
        gene_num_threshold=args.gene_num_threshold,
        normalization=args.normalization,
        log1p=args.log1p
    )

    # Optionally output the processed data
    # adata.write('output.h5ad')  # Save the processed data
    print(adata)


