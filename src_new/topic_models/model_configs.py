
class TopicConfigs:
    def __init__(self):
        self.gene_size = 1000
        self.act = 'relu'
        self.enc_drop = 0.5
        self.device = 'cuda'
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6
        self.clip = 0
        self.verbose = 1


class LDAconfigs:
    def __init__(self):
        self.alpha = 0.1
        self.beta = 0.1
        self.iterations= 20
        self.device = 'cuda'
        self.verbose = 1
        self.batch_size = 128