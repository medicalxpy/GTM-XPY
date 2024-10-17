import argparse
import torch
import pickle
import scanpy as sc
import pandas as pd


from pathlib import Path
from scipy.sparse import issparse
from topic_models import GETM,GLDA
from configs import TopicConfigs,LDAconfigs
from h5ad_preprocessing import H5adPreprocessing
device = torch.device("cuda")

parser = argparse.ArgumentParser(description='train Topic Model using gene embedding/gene counts')

parser.add_argument('--mode', type=str, default='train', 
                            help='train or continue model')
parser.add_argument('--data_type',type=str,default='gene_embedding',
                            help='you can use gene embedding or gene counts to train topic model')
parser.add_argument('--num_topics',type=int,default="200",
                            help='number of topics')
parser.add_argument('--counts_path', type=Path, default='../data/PBMC.h5ad',
                        help='Path to directory containing  gene counts(.h5ad).')
parser.add_argument('--embedding_path', type=Path, default='../data/embeddings/gene.pkl',
                        help='Path to directory containing  gene embedding(.pkl).')
parser.add_argument('--output_directory', type=str, default='../data/matrix/',
                        help='Path to directory where cell x topic and gene x topic matrix will be saved.')
parser.add_argument('--num_epoch',type= int, default= 10 ,
                        help='number of train epoch')
parser.add_argument('--model_name', type=str, default='gene_PBMC.pth',
                        help='Path to directory where cell x topic and gene x topic matrix will be saved.')
parser.add_argument('--load_path', type=Path, default='../save_model/Geneformer-blood_0.pth',
                        help='Path to directory where last model saved.')



args = parser.parse_args()



if __name__ == '__main__':


    if args.mode == "train":

        if args.data_type =="gene_counts":
            adata = sc.read_h5ad(args.counts_path)
            model = GLDA(num_topics=args.num_topics,args=LDAconfigs())
            model.fit(adata)
            theta = model.get_cell_topic_distribution().detach().cpu().numpy()
            phi = model.get_topic_gene_distribution().detach().cpu().numpy()
            pd.DataFrame(theta, index=adata.obs_names).to_csv('{args.output_directory}/cell x topics_LDA.csv')
            pd.DataFrame(phi, columns=adata.var_names).to_csv('{args.output_directory}/topics x gene_LDA.csv')




        if args.data_type == "gene_embedding":
            # load data
            with open(args.embedding_path, 'rb') as f:
                gene_embeddings = pickle.load(f)
                gene_embeddings.set_index(gene_embeddings.columns[0], inplace=True)
                gene_embeddings = gene_embeddings.groupby(gene_embeddings.index).mean()
            gene_embeddings_genes = gene_embeddings.index
            adata = H5adPreprocessing(args.counts_path)
            #adata=sc.read_h5ad(args.counts_path)
            print('成功读取数据')
            gene_data_genes = adata.var_names
            common_genes = gene_embeddings_genes.intersection(gene_data_genes)
            gene_counts_common = adata[:, common_genes]
            gene_embeddings_common = gene_embeddings.loc[common_genes].to_numpy().astype('float32')
            gene_embeddings = torch.tensor(gene_embeddings_common, dtype=torch.float32).to(device)
            print(f'共同基因数: {len(common_genes)}')


            model = GETM(embeddings=gene_embeddings, num_topics=args.num_topics, args=TopicConfigs())
            num_epochs= args.num_epoch
            for epoch in range(num_epochs):
                model.train_one_epoch(epoch, gene_counts_common)
                torch.cuda.empty_cache()
            # save model
            torch.save(model.state_dict(), f'../save_model/{args.model_name}.pth')
            print("成功保存模型：",args.model_name)
            
            # conut topic embedding
            beta = model.get_beta().to(device)
            topic_embeddings = torch.matmul(beta.to(device), gene_embeddings.to(device))

            # save topic embedding
            with open(f'../data/topic_embedding/{args.model_name}.pkl', 'wb') as f:


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


            pd.DataFrame(single_cell_topics_matrix, index=gene_counts_common.obs_names).to_csv(f'{args.output_directory}/cell_x_topics_{args.model_name}.csv')
            pd.DataFrame(beta, columns=common_genes).to_csv(f'{args.output_directory}/topics_x_gene_{args.model_name}.csv')

    if args.mode == "continue":
        # if args.data_type =="gene_counts":
        #     adata = sc.read_h5ad(args.counts_path)
        #     model = GLDA(num_topics=args.num_topics,args=LDAconfigs())
        #     model.fit(adata)
        #     theta = model.get_cell_topic_distribution().detach().cpu().numpy()
        #     phi = model.get_topic_gene_distribution().detach().cpu().numpy()
        #     pd.DataFrame(theta, index=adata.obs_names).to_csv('{args.output_directory}/cell x topics_LDA.csv')
        #     pd.DataFrame(phi, columns=adata.var_names).to_csv('{args.output_directory}/topics x gene_LDA.csv')




        if args.data_type == "gene_embedding":
             # load data
            with open(args.embedding_path, 'rb') as f:
                gene_embeddings = pickle.load(f)
                gene_embeddings.set_index(gene_embeddings.columns[0], inplace=True)
                gene_embeddings = gene_embeddings.groupby(gene_embeddings.index).mean()
            gene_embeddings_genes = gene_embeddings.index
            adata = H5adPreprocessing(args.counts_path)
            #adata=sc.read_h5ad(args.counts_path)
            print('成功读取数据')
            gene_data_genes = adata.var_names
            common_genes = gene_embeddings_genes.intersection(gene_data_genes)
            gene_counts_common = adata[:, common_genes]
            gene_embeddings_common = gene_embeddings.loc[common_genes].to_numpy().astype('float32')
            gene_embeddings = torch.tensor(gene_embeddings_common, dtype=torch.float32).to(device)
            print(f'共同基因数: {len(common_genes)}')


            model = GETM(embeddings=gene_embeddings, num_topics=args.num_topics, args=TopicConfigs())
            model.load_state_dict(torch.load(args.load_path))
            model.to(device)
            num_epochs= args.num_epoch
            for epoch in range(num_epochs):
                model.train_one_epoch(epoch, gene_counts_common)
                torch.cuda.empty_cache()
            # save model
            torch.save(model.state_dict(), f'../save_model/{args.model_name}.pth')
            print("成功保存模型：",args.model_name)
            
            # conut topic embedding
            beta = model.get_beta().to(device)
            topic_embeddings = torch.matmul(beta.to(device), gene_embeddings.to(device))

            # save topic embedding
            with open(f'../data/topic_embedding/{args.model_name}.pkl', 'wb') as f:


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


            pd.DataFrame(single_cell_topics_matrix, index=gene_counts_common.obs_names).to_csv(f'{args.output_directory}/cell_x_topics_{args.model_name}.csv')
            pd.DataFrame(beta, columns=common_genes).to_csv(f'{args.output_directory}/topics_x_gene_{args.model_name}.csv')