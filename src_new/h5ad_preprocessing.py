#!/usr/bin/env python3

import scanpy as sc
import scipy.sparse as sp
import argparse
import pandas as pd
import re
import numpy as np

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

def preprocess_h5ad(file_input_path,
                    cell_ge_expr_threshold=1000,
                    gene_num_threshold=60,
                    normalization=True,
                    log1p=False,
                    cell_type_threshold=100,
                    cell_type_label='cell_type'):
    """
    Preprocess the AnnData object from an h5ad file.

    Parameters:
    - file_input_path: Path to the input h5ad file.
    - cell_ge_expr_threshold: Threshold for cell gene expression.
    - gene_num_threshold: Minimum number of genes per cell.
    - normalization: Whether to perform normalization.
    - log1p: Whether to apply log1p transformation.
    - cell_type_threshold: Threshold for excluding rare cell types.
    - cell_type_label: Column name for cell types in obs.

    Returns:
    - Preprocessed AnnData object.
    """
    adata = sc.read_h5ad(file_input_path)

    # Filter the cells based on gene count threshold
    sc.pp.filter_cells(adata, min_genes=gene_num_threshold)

    # Filter the cells based on cell type threshold
    cell_types = adata.obs[cell_type_label]
    type_num = cell_types.value_counts()
    adata = adata[adata.obs[cell_type_label].isin(type_num[type_num >= cell_type_threshold].index)]

    # Create a balanced subset
    balanced_adata = sc.AnnData(X=np.empty((0, adata.n_vars)))

    for cell_type, count in type_num.items():
        if count >= 200:
            sample = adata[adata.obs[cell_type_label] == cell_type].sample(n=200, random_state=42)
        else:
            sample = adata[adata.obs[cell_type_label] == cell_type]
        balanced_adata = balanced_adata.concatenate(sample)

    # Screen genes
    gene_means = balanced_adata.X.mean(axis=0)
    gene_vars = balanced_adata.X.var(axis=0)
    valid_genes = (gene_means >= 10) & (gene_vars / gene_means > 1)
    balanced_adata = balanced_adata[:, valid_genes]

    # Apply log1p transformation if specified
    if log1p:
        sc.pp.log1p(balanced_adata)

    # Perform normalization if specified
    if normalization:
        sc.pp.normalize_total(balanced_adata, target_sum=1)

    # Find and update the ensemble_id column
    for index in balanced_adata.var.columns:
        if check_if_all_strings(balanced_adata.var[index]):
            if balanced_adata.var[index].head(10).str.startswith('ENSG').all():
                balanced_adata.var[index] = balanced_adata.var[index].apply(remove_version_number)
                balanced_adata.var_names = balanced_adata.var[index]
                break

    return balanced_adata

def main():
    parser = argparse.ArgumentParser(description="Preprocess an h5ad file.")
    parser.add_argument("file_input_path", help="Path to the input h5ad file.")
    parser.add_argument("--cell_ge_expr_threshold", type=int, default=1000, help="Threshold for cell gene expression.")
    parser.add_argument("--gene_num_threshold", type=int, default=60, help="Minimum number of genes per cell.")
    parser.add_argument("--normalization", action="store_true", help="Whether to perform normalization.")
    parser.add_argument("--log1p", action="store_true", help="Whether to apply log1p transformation.")
    parser.add_argument("--cell_type_threshold", type=int, default=100, help="Threshold for excluding rare cell types.")
    parser.add_argument("--cell_type_label", default='cell_type', help="Column name for cell types in obs.")

    args = parser.parse_args()

    preprocessed_adata = preprocess_h5ad(args.file_input_path,
                                         cell_ge_expr_threshold=args.cell_ge_expr_threshold,
                                         gene_num_threshold=args.gene_num_threshold,
                                         normalization=args.normalization,
                                         log1p=args.log1p,
                                         cell_type_threshold=args.cell_type_threshold,
                                         cell_type_label=args.cell_type_label)

    # Write the preprocessed data back to a new h5ad file
    output_file_path = f"{args.file_input_path}_preprocessed.h5ad"
    preprocessed_adata.write(output_file_path)
    print(f"Preprocessed data written to {output_file_path}.")

if __name__ == "__main__":
    main()