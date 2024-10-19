import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
from scipy.sparse import csr_matrix
from configs import LDAconfigs

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return F.softmax(self.fc_out(h), dim=1)

class GLDA:
    def __init__(self, num_topics=int, args=LDAconfigs, sample_way='Gibbs'):
        self.n_topics = num_topics
        self.alpha = args.alpha
        self.beta = args.beta
        self.iterations = args.iterations
        self.device = args.device
        self.batch_size = args.batch_size 
        self.theta = None  # Cell-topic distribution
        self.phi = None  # Topic-gene distribution
        self.sample_way = sample_way
        self.verbose = args.verbose
        self.encoder = None
        self.decoder = None

    def initialize_vae(self, input_dim, hidden_dim, latent_dim, output_dim):
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.encoder.to(self.device)
        self.decoder.to(self.device)

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
        X = self._process_input(adata.X)
        D, V = X.shape
        cell_topic_count, topic_gene_count, topic_count = self.initialize_parameters(D, V)

        if self.sample_way == 'VAE':
            self.initialize_vae(input_dim=X.shape[1], hidden_dim=256, latent_dim=self.n_topics, output_dim=X.shape[1])
            optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.001)

        for it in range(self.iterations):
            for i in range(0, D, self.batch_size):
                batch_X = X[i:i + self.batch_size, :]
                batch_size = batch_X.shape[0]

                if self.sample_way == 'VAE':
                    mu, logvar = self.encoder(batch_X)
                    z = self.reparameterize(mu, logvar)
                    theta = self.decoder(z)

                    recon_loss = F.binary_cross_entropy(theta, batch_X, reduction='sum')
                    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                    loss = recon_loss + kl_divergence
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # update
                    cell_topic_count[i:i + batch_size, :] += theta.detach()
                    topic_gene_count += torch.bmm(theta.unsqueeze(1), batch_X.unsqueeze(2)).squeeze().t()
                    topic_count += torch.sum(theta, dim=0)

                    # Print message 
                    print('该batch已完成')

                elif self.sample_way == 'Gibbs':
                    # Gibbs sampling code here...
                    pass

                elif self.sample_way == 'MH':
                    # Metropolis-Hastings code here...
                    pass

            if self.verbose and it % 10 == 0:
                print(f"Iteration {it + 1}/{self.iterations}")

        self.theta = (cell_topic_count + self.alpha) / \
                     (torch.sum(cell_topic_count, axis=1, keepdims=True) + self.n_topics * self.alpha)
        self.phi = (topic_gene_count + self.beta) / \
                   (torch.sum(topic_gene_count, axis=1, keepdims=True) + V * self.beta)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_cell_topic_distribution(self):
        return self.theta

    def get_topic_gene_distribution(self):
        return self.phi