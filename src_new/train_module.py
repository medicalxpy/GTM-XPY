import argparse
import torch
import pickle
import scanpy as sc
import pandas as pd


from pathlib import Path
from scipy.sparse import issparse
from configs import TopicConfigs,LDAconfigs
from topic_models import GETM,GLDA
from src_new.embedding_module import GeneProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


configs=TopicConfigs
parser = argparse.ArgumentParser(description='train Topic Model using gene embedding/gene counts')

parser.add_argument('--mode', type=str, default='train', 
                            help='train or eval model')
parser.add_argument('--data_type',type=str,default='gene embedding',
                            help='you can use gene embedding or gene counts to train topic model')
parser.add_argument('--num_topics',type=int,default="100",
                            help='number of topics')
parser.add_argument('--counts_path', type=Path, default='../data/PBMC.h5ad',
                        help='Path to directory containing  gene counts(.h5ad).')
parser.add_argument('--embedding_path', type=Path, default='../data/embeddings/gene.pkl',
                        help='Path to directory containing  gene embedding(.pkl).')
parser.add_argument('--output_directory', type=Path, default='../data/matrix/',
                        help='Path to directory where cell x topic and gene x topic matrix will be saved.')
parser.add_argument('--num_epoch',type= int, default= 10 ,
                        help='number of train epoch')
args = parser.parse_args()



if __name__ == '__main__':




    if args.data_type =="gene counts":
        adata = sc.read_h5ad("{args.counts_path}")
        model = GLDA(num_topics=args.num_topics,args=LDAconfigs)
        model.fit(adata)
        theta = model.get_cell_topic_distribution().detach().cpu().numpy()
        phi = model.get_topic_gene_distribution().detach().cpu().numpy()
        pd.DataFrame(theta, index=adata.obs_names).to_csv('{args.output_directory}/cell x topics_LDA.csv')
        pd.DataFrame(phi, columns=adata.var_names).to_csv('{args.output_directory}/topics x gene_LDA.csv')




    if args.data_type == "gene embedding":
        # load data
        with open('{args.embedding_path}', 'rb') as f:
            gene_embeddings = pickle.load(f)
        adata = sc.read_h5ad('{args.counts_path}')
        gene_embeddings_genes = gene_embeddings.index
        gene_data_genes = adata.var_names
        common_genes = gene_embeddings_genes.intersection(gene_data_genes)
        gene_counts_common = adata[:, common_genes]
        gene_embeddings_common = gene_embeddings.loc[common_genes].to_numpy()
        gene_embeddings = torch.tensor(gene_embeddings_common, dtype=torch.float32).to(device)

        model = GETM(embeddings=gene_embeddings, num_topics=args.num_epoch, args=args)
        num_epochs= args.num_epoch
        for epoch in range(num_epochs):
            model.train_one_epoch(epoch, gene_counts_common)
        # save model
        torch.save(model.state_dict(), '../save_model/GTM.pth')
        beta = model.get_beta().to(device)
        # conut topic embedding
        topic_embeddings = torch.matmul(beta.to(device), gene_embeddings.to(device))

        # save topic embedding
        with open('../topic embedding/topic.pkl', 'wb') as f:
            pickle.dump(topic_embeddings, f)
        
        # Single Cell-Topics Matrix
        sc.pp.normalize_total(gene_counts_common, target_sum=1) 
        if issparse(gene_counts_common.X):
            gene_counts_tensor = torch.tensor(gene_counts_common.X.toarray(), dtype=torch.float32).to(device)
        else:
            gene_counts_tensor = torch.tensor(gene_counts_common.X, dtype=torch.float32).to(device)

        theta, _ = model.get_theta(gene_counts_tensor)  
        single_cell_topics_matrix = theta.detach().cpu().numpy() 

        # Topic-Gene Similarity Matrix
        beta = model.get_beta().detach().cpu().numpy()  


        pd.DataFrame(single_cell_topics_matrix, index=gene_counts_common.obs_names).to_csv('{args.output_directory}/cell x topics_GETM.csv')
        pd.DataFrame(beta, columns=common_genes).to_csv('{args.output_directory}/topics x gene_GETM.csv')

