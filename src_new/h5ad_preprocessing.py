import scanpy as sc
import scipy.sparse as sp  
import argparse

def h5ad_preprocessing(file_input_path,
                       cell_ge_expr_threshold=1000,
                       gene_num_threshold=60,
                       normalization=True,
                       log1p=False):

    """
    Raw data preprocessing.
    
    Args:
        file_input_path (str): Path to the input .h5ad file.
        cell_ge_expr_threshold (int): Minimum total gene expression count for a cell.
        gene_num_threshold (int): Minimum number of genes detected in a cell.
        normalization (bool): Whether to perform total-count normalization.
        log1p (bool): Whether to perform log1p transformation.
    """
    
    if file_input_path.endswith('.h5ad'):
        adata = sc.read_h5ad(file_input_path)
        
        # Get the raw data
        if 'raw_counts' in adata.layers:
            adata.X = adata.layers['raw_counts']
        else:
            raise ValueError("Layer 'raw_counts' not found in adata.layers.")
        
        # Filter the cells
        # Knock out cells with a low number of genes
        sc.pp.filter_cells(adata, min_genes=gene_num_threshold)
        
        # Remove cells with low number of gene expressions
        if sp.issparse(adata.X):
            adata.obs['n_count'] = adata.X.sum(axis=1).A1
        else:
            adata.obs['n_count'] = adata.X.sum(axis=1)
        
        valid_cells = adata.obs['n_count'] >= cell_ge_expr_threshold
        adata = adata[valid_cells, :]
        
        # log1p conversion
        if log1p:
            sc.pp.log1p(adata)
    
        # normalization
        if normalization:
            sc.pp.normalize_total(adata, target_sum=1)  
        
        # Return the data
        return adata
    else:
        return 'This is not an h5ad file'

def main():
    parser = argparse.ArgumentParser(description='Preprocess h5ad files.')
    parser.add_argument('--input', type=str, help='Path to the input .h5ad file.')
    parser.add_argument('--output', type=str, default='preprocessed_data.h5ad', help='Output path for the preprocessed data.')
    parser.add_argument('--cell-ge-expr-threshold', type=int, default=1000, help='Minimum total gene expression count for a cell.')
    parser.add_argument('--gene-num-threshold', type=int, default=60, help='Minimum number of genes detected in a cell.')
    parser.add_argument('--no-normalization', action='store_false', help='Do not perform total-count normalization.')
    parser.add_argument('--no-log1p', action='store_false', help='Do not perform log1p transformation.')

    args = parser.parse_args()

    # Preprocess the data
    adata = h5ad_preprocessing(args.input,
                               cell_ge_expr_threshold=args.cell_ge_expr_threshold,
                               gene_num_threshold=args.gene_num_threshold,
                               normalization=args.no_normalization,
                               log1p=args.no_log1p)
    
    # Save the preprocessed data
    if isinstance(adata, str):
        print(adata)
    else:
        adata.write(args.output)
        print(f'Preprocessed data saved to {args.output}')

if __name__ == '__main__':
    main()


