import torch
import scanpy as sc
from scipy.sparse import csr_matrix
from configs import LDAconfigs

class GLDA:
    def __init__(self, num_topics=int, args=LDAconfigs):
        self.n_topics = num_topics
        self.alpha = args.alpha
        self.beta = args.beta
        self.iterations = args.iterations
        self.device = args.device
        self.batch_size = args.batch_size 
        self.theta = None  # Cell-topic distribution
        self.phi = None  # Topic-gene distribution

    def _process_input(self, X):
        if isinstance(X, csr_matrix):
            X = X.toarray()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        return X

    def initialize_parameters(self, D, V):
        cell_topic_count = torch.zeros((D, self.n_topics), dtype=torch.float32, device=self.device)
        topic_gene_count = torch.zeros((self.n_topics, V), dtype=torch.float32, device=self.device)
        topic_count = torch.zeros(self.n_topics, dtype=torch.float32, device=self.device)
        return cell_topic_count, topic_gene_count, topic_count

    def fit(self, adata):
        sc.pp.filter_genes(adata, min_cells=10)
        sc.pp.filter_cells(adata, min_genes=200)
        X = self._process_input(adata.X)
        D, V = X.shape
        cell_topic_count, topic_gene_count, topic_count = self.initialize_parameters(D, V)

        for it in range(self.iterations):

            for i in range(0, D, self.batch_size):
                batch_X = X[i:i + self.batch_size, :]  
                batch_size = batch_X.shape[0]


                batch_cell_topic_count = torch.zeros((batch_size, self.n_topics), device=self.device)


                non_zero_indices = (batch_X > 0).nonzero(as_tuple=True)
                topics = torch.randint(0, self.n_topics, (len(non_zero_indices[0]),), device=self.device)

                for idx, (d, v) in enumerate(zip(*non_zero_indices)):
                    topic = topics[idx]
                    batch_cell_topic_count[d, topic] += 1
                    topic_gene_count[topic, v] += 1
                    topic_count[topic] += 1

                    topic_probs = (batch_cell_topic_count[d] + self.alpha) * \
                                  (topic_gene_count[:, v] + self.beta) / \
                                  (topic_count + V * self.beta)

                    topic_probs = torch.clamp(topic_probs, min=1e-10)
                    topic_probs /= torch.sum(topic_probs)

                    new_topic = torch.multinomial(topic_probs, 1).item()

                    batch_cell_topic_count[d, new_topic] += 1
                    topic_gene_count[new_topic, v] += 1
                    topic_count[new_topic] += 1

            if self.verbose and it % 10 == 0:
                print(f"Iteration {it + 1}/{self.iterations}")

        self.theta = (cell_topic_count + self.alpha) / \
                     (torch.sum(cell_topic_count, axis=1, keepdims=True) + self.n_topics * self.alpha)
        self.phi = (topic_gene_count + self.beta) / \
                   (torch.sum(topic_gene_count, axis=1, keepdims=True) + V * self.beta)

    def get_cell_topic_distribution(self):
        return self.theta

    def get_topic_gene_distribution(self):
        return self.phi
