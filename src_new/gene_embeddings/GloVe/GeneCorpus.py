#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from scipy.sparse import csr_matrix

class GeneCorpus(object):
    """
    Class for constructing a gene co-occurrence matrix
    from gene expression data.
    
    A dictionary mapping gene indices to gene names can optionally
    be supplied. If left None, it will be constructed
    from the data.
    """

    def __init__(self, gene_dictionary=None):
        self.gene_dictionary = {}
        self.gene_dictionary_supplied = False
        self.matrix = None

        if gene_dictionary is not None:
            self._check_dict(gene_dictionary)
            self.gene_dictionary = gene_dictionary
            self.gene_dictionary_supplied = True

    def _check_dict(self, gene_dictionary):
        if (np.max(list(gene_dictionary.values())) != (len(gene_dictionary) - 1)):
            raise Exception('The largest id in the dictionary '
                            'should be equal to its length minus one.')

        if np.min(list(gene_dictionary.values())) != 0:
            raise Exception('Dictionary ids should start at zero')

    def fit(self, expression_data, window=10):
        """
        Perform a pass through the expression data to construct
        the co-occurrence matrix.

        Parameters:
        - expression_data: sparse matrix where rows represent cells
        and columns represent genes.
        - int window: the length of the context window used for co-occurrence.
        """

        self.matrix = self.construct_cooccurrence_matrix(expression_data, window)

    def construct_cooccurrence_matrix(self, expression_data, window):
        """
        Construct the co-occurrence matrix from the gene expression data.

        Parameters:
        - expression_data: sparse matrix of gene expressions
        - int window: context window size for co-occurrence.

        Returns:
        - coo_matrix: Sparse matrix representing gene co-occurrences.
        """
        num_genes = expression_data.shape[1]
        rows, cols, data = [], [], []

        for i in range(expression_data.shape[0]):  # Iterate over each cell
            non_zero_indices = expression_data[i, :].nonzero()[1]
            num_non_zero = len(non_zero_indices)

            for j in range(num_non_zero):
                for k in range(j + 1, min(j + 1 + window, num_non_zero)):
                    gene_i = non_zero_indices[j]
                    gene_j = non_zero_indices[k]

                    # Increment the co-occurrence count
                    rows.append(gene_i)
                    cols.append(gene_j)
                    data.append(1)  # Simple count, can be adjusted if needed

                    # Symmetric entry
                    rows.append(gene_j)
                    cols.append(gene_i)
                    data.append(1)

        return csr_matrix((data, (rows, cols)), shape=(num_genes, num_genes))

    def save(self, filename):
        with open(filename, 'wb') as savefile:
            pickle.dump((self.gene_dictionary, self.matrix),
                        savefile,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        instance = cls()
        with open(filename, 'rb') as savefile:
            instance.gene_dictionary, instance.matrix = pickle.load(savefile)
        return instance




