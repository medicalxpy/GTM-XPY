#!/usr/bin/env python
# coding: utf-8


import array
import collections
import io
import pickle
import numpy as np
import scipy.sparse as sp
import numbers

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance."""
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


class GeneGloVe(object):
    """
    Class for estimating GloVe embeddings using the
    gene co-occurrence matrix.
    """
    def read_data(self):
        """Read gene expression data from .h5ad file."""
        adata = anndata.read_h5ad(self.h5ad_file)
        return adata.X, adata.var_names
    
    def __init__(self, no_components=30, learning_rate=0.05,
                 alpha=0.75, max_count=100, max_loss=10.0,
                 random_state=None):
        """
        Parameters:
        - int no_components: number of latent dimensions
        - float learning_rate: learning rate for SGD estimation.
        - float alpha, float max_count: parameters for the
          weighting function (see the GloVe paper).
        - float max_loss: the maximum absolute value of calculated
                          gradient for any single co-occurrence pair.
        - random_state: random state used to initialize optimization
        """

        self.no_components = no_components
        self.learning_rate = float(learning_rate)
        self.alpha = float(alpha)
        self.max_count = float(max_count)
        self.max_loss = max_loss

        self.gene_vectors = None
        self.gene_biases = None

        self.vectors_sum_gradients = None
        self.biases_sum_gradients = None

        self.gene_dictionary = None
        self.inverse_dictionary = None

        self.random_state = random_state

    def fit(self, matrix, epochs=5, no_threads=2, verbose=False):
        """
        Estimate the gene embeddings.

        Parameters:
        - scipy.sparse.coo_matrix matrix: co-occurrence matrix
        - int epochs: number of training epochs
        - int no_threads: number of training threads
        - bool verbose: print progress messages if True
        """

        shape = matrix.shape

        if (len(shape) != 2 or
            shape[0] != shape[1]):
            raise Exception('Co-occurrence matrix must be square')

        if not sp.isspmatrix_coo(matrix):
            raise Exception('Co-occurrence matrix must be in the COO format')

        random_state = check_random_state(self.random_state)
        self.gene_vectors = ((random_state.rand(shape[0],
                                                self.no_components) - 0.5)
                             / self.no_components)
        self.gene_biases = np.zeros(shape[0], dtype=np.float64)

        self.vectors_sum_gradients = np.ones_like(self.gene_vectors)
        self.biases_sum_gradients = np.ones_like(self.gene_biases)

        shuffle_indices = np.arange(matrix.nnz, dtype=np.int32)

        if verbose:
            print('Performing %s training epochs '
                  'with %s threads' % (epochs, no_threads))

        for epoch in range(epochs):
            if verbose:
                print('Epoch %s' % epoch)

            # Shuffle the co-occurrence matrix
            random_state.shuffle(shuffle_indices)

            # Fit the vectors (this function needs to be implemented)
            fit_vectors(self.gene_vectors,
                        self.vectors_sum_gradients,
                        self.gene_biases,
                        self.biases_sum_gradients,
                        matrix.row,
                        matrix.col,
                        matrix.data,
                        shuffle_indices,
                        self.learning_rate,
                        self.max_count,
                        self.alpha,
                        self.max_loss,
                        int(no_threads))

            if not np.isfinite(self.gene_vectors).all():
                raise Exception('Non-finite values in gene vectors. '
                                'Try reducing the learning rate or the '
                                'max_loss parameter.')

    def add_dictionary(self, dictionary):
        """
        Supply a gene-id dictionary to allow similarity queries.
        """
        if self.gene_vectors is None:
            raise Exception('Model must be fit before adding a dictionary')

        if len(dictionary) > self.gene_vectors.shape[0]:
            raise Exception('Dictionary length must be smaller '
                            'or equal to the number of gene vectors')

        self.gene_dictionary = dictionary
        self.inverse_dictionary = {v: k for k, v in dictionary.items()}

    def save(self, filename):
        """Serialize model to filename."""
        with open(filename, 'wb') as savefile:
            pickle.dump(self.__dict__, savefile, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        """Load model from filename."""
        instance = cls()
        with open(filename, 'rb') as savefile:
            instance.__dict__ = pickle.load(savefile)
        return instance

    def most_similar(self, gene, number=5):
        """
        Run a similarity query, retrieving number
        most similar genes.
        """
        if self.gene_vectors is None:
            raise Exception('Model must be fit before querying')

        if self.gene_dictionary is None:
            raise Exception('No gene dictionary supplied')

        try:
            gene_idx = self.gene_dictionary[gene]
        except KeyError:
            raise Exception('Gene not in dictionary')

        return self._similarity_query(self.gene_vectors[gene_idx], number)

    def _similarity_query(self, gene_vec, number):
        """Find similar genes based on the gene vector."""
        dst = (np.dot(self.gene_vectors, gene_vec)
               / np.linalg.norm(self.gene_vectors, axis=1)
               / np.linalg.norm(gene_vec))
        gene_ids = np.argsort(-dst)

        return [(self.inverse_dictionary[x], dst[x]) for x in gene_ids[:number]
                if x in self.inverse_dictionary]







