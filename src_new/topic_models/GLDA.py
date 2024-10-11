import torch
from scipy.sparse import csr_matrix
from configs import LDAconfigs

class GLDA:
    def __init__(self, num_topics=int, args=LDAconfigs):
        """
        LDA model initialization
        
        Parameters
        ----------
        num_topics: int 
                Number of topics
        alpha: 
                Dirichlet parameter for cell-topic distribution
        beta: 
                Dirichlet parameter for topic-gene distribution
        iterations: 
                Number of Gibbs sampling iterations
        device: 
                Compute device ('cuda' for GPU or 'cpu')
        """
        self.n_topics = num_topics
        self.alpha = args.alpha
        self.beta = args.beta
        self.iterations = args.iterations
        self.device = args.device
        self.theta = None  # Cell-topic distribution
        self.phi = None  # Topic-gene distribution

    def _process_input(self, X):
        """
        Process the input data to convert it into a GPU-compatible format
        :param X: Input matrix (cells x genes)
        :return: Processed matrix in GPU tensor format (D x V)
        """
        # Convert sparse matrices to dense format
        if isinstance(X, csr_matrix):
            X = X.toarray()
        # Convert the matrix to a PyTorch tensor and move it to the specified device (GPU or CPU)
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        return X

    def initialize_parameters(self, X):
        """
        Initialize model parameters
        :param X: Input gene frequency matrix (D x V)
        :return: Initialized cell-topic and topic-gene count matrices
        """
        D, V = X.shape

        # Randomly assign a topic to each gene and initialize count matrices
        cell_topic_count = torch.zeros((D, self.n_topics), dtype=torch.float32, device=self.device)
        topic_gene_count = torch.zeros((self.n_topics, V), dtype=torch.float32, device=self.device)
        topic_count = torch.zeros(self.n_topics, dtype=torch.float32, device=self.device)

        # Randomly assign topics and update count matrices
        for d in range(D):
            for v in range(V):
                if X[d, v] > 0:
                    for _ in range(int(X[d, v].item())):
                        topic = torch.randint(0, self.n_topics, (1,), device=self.device).item()
                        cell_topic_count[d, topic] += 1
                        topic_gene_count[topic, v] += 1
                        topic_count[topic] += 1

        return cell_topic_count, topic_gene_count, topic_count

    def fit(self, adata):
        """
        Train LDA using Gibbs sampling
        """
        X = self._process_input(adata.X)
        D, V = X.shape
        cell_topic_count, topic_gene_count, topic_count = self.initialize_parameters(X)

        for it in range(self.iterations):
            for d in range(D):
                for v in range(V):
                    if X[d, v] > 0:
                        for _ in range(int(X[d, v].item())):
                            current_topic = torch.randint(0, self.n_topics, (1,), device=self.device).item()

                            cell_topic_count[d, current_topic] = torch.clamp(cell_topic_count[d, current_topic] - 1, min=0)
                            topic_gene_count[current_topic, v] = torch.clamp(topic_gene_count[current_topic, v] - 1, min=0)
                            topic_count[current_topic] = torch.clamp(topic_count[current_topic] - 1, min=0)

                            topic_probs = (cell_topic_count[d] + self.alpha) * \
                                          (topic_gene_count[:, v] + self.beta) / \
                                          torch.clamp(topic_count + V * self.beta, min=1e-10)

                            # Normalize probabilities
                            topic_probs /= torch.sum(topic_probs)

                            if torch.isnan(topic_probs).any() or torch.isinf(topic_probs).any() or (topic_probs < 0).any():
                                print(f"Invalid probabilities found: cell index {d}, gene index {v}")
                                topic_probs = torch.ones_like(topic_probs) / len(topic_probs)

                            new_topic = torch.multinomial(topic_probs, 1).item()

                            # Update
                            cell_topic_count[d, new_topic] += 1
                            topic_gene_count[new_topic, v] += 1
                            topic_count[new_topic] += 1

            if self.verbose:
                log_likelihood = self.compute_log_likelihood(X, cell_topic_count, topic_gene_count, topic_count)
                perplexity = self.compute_perplexity(X, cell_topic_count, topic_gene_count, topic_count)
                print(f"Iteration {it + 1}/{self.iterations}, Log-likelihood: {log_likelihood}, Perplexity: {perplexity}")

        self.theta = (cell_topic_count + self.alpha) / \
                     (torch.sum(cell_topic_count, axis=1, keepdims=True) + self.n_topics * self.alpha)
        self.phi = (topic_gene_count + self.beta) / \
                   (torch.sum(topic_gene_count, axis=1, keepdims=True) + V * self.beta)

    def compute_perplexity(self, X, cell_topic_count, topic_gene_count, topic_count):
        """
        Compute Perplexity
        :param X: Input gene frequency matrix (D x V)
        :param cell_topic_count: Cell-topic count matrix
        :param topic_gene_count: Topic-gene count matrix
        :param topic_count: Topic count vector
        :return: Perplexity value
        """
        D, V = X.shape
        log_perplexity = 0.0
        total_genes = torch.sum(X)
        for d in range(D):
            for v in range(V):
                if X[d, v] > 0:
                    prob = 0.0
                    for k in range(self.n_topics):
                        prob += (cell_topic_count[d, k] + self.alpha) * \
                                (topic_gene_count[k, v] + self.beta) / \
                                (topic_count[k] + V * self.beta)
                    log_perplexity -= X[d, v] * torch.log(prob)
        return torch.exp(log_perplexity / total_genes).item()

    def compute_log_likelihood(self, X, cell_topic_count, topic_gene_count, topic_count):
        D, V = X.shape
        log_likelihood = 0.0
        for d in range(D):
            for v in range(V):
                if X[d, v] > 0:
                    prob = 0.0
                    for k in range(self.n_topics):
                        prob += (cell_topic_count[d, k] + self.alpha) * \
                                (topic_gene_count[k, v] + self.beta) / \
                                (topic_count[k] + V * self.beta)
                log_likelihood += X[d, v] * torch.log(prob)
        return log_likelihood.item()

    def get_cell_topic_distribution(self):
        """
        Return the cell-topic distribution matrix (Theta)
        :return: (D x K) cell-topic distribution matrix
        """
        return self.theta

    def get_topic_gene_distribution(self):
        """
        Return the topic-gene distribution matrix (Phi)
        :return: (K x V) Topic-gene distribution matrix
        """
        return self.phi
