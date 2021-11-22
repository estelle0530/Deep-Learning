# Multimodal MARS: Representational Learning of Multi-omic Single Cell Datasets to Classify Diseased States

## Background
With emerging multimodal single cell sequencing techniques, there has been increasing research on integrative modeling of multimodal data. 
Particularly, variational autoencoder based methods have been applied to multi-modal single cell clustering analysis with various priors to accommodate data types. 
Although most of these proposed multi-modal models present interpretable clustering results, cell type annotations still largely rely on manual work by matching signature genes to each cluster. The dearth of published work on automatic cell type annotation by leveraging existing annotated transcriptomic data to annotate chromatin accessibility data calls for an integrative method to extend current transcriptome annotation techniques to multimodal data. 

## Method
We integrated MARS framework and VAE-based multimodal co-embedding to enable multimodal cell type annotations. We named this multimodal MARS cell annotation algorithm MM-MARS. MM-MARS is built upon the semi-supervised cell type annotation meta-learning model from MARS by modifying the co-embedding function from an autoencoder to a variational autoencoder framework. MM-MARS aims to optimize the co-embedding for multi-omics data to recover the latent structure and further leverage existing annotated transcriptome sets to annotate ATAC-seq from different experiments.

Motivated by the dearth of research on integrating multi-omic data for cell type annotation, we aim to improve representational learning for both scRNA-seq and scATAC-seq and improve cell type annotation accuracy and discovery with our MM-MARS model. We will then apply this model to studying tumoral heterogeneity in melanoma, and construct cell type specific GRNs to discover mechanisms driving acquired drug resistance of pembrolizumab.

![alt text](https://github.com/estelleyao0530/Deep-Learning/blob/main/Figure/mars_schematic.png)

## Results
1. We demonstrated that MARS can be applied to annotate multimodal datasets as MM-MARS embeddings and distribution of MM-MARS labels are largely biologically meaningful
2. Our Variational autoencoder based MM-MARS outperforms autoencoder based MM-MARS in multimodal cell type annotation 
3. Co-embedding of scRNA and scATAC in pretrainng improves characterization cell types in VAE and CVAE MM-MARS
4. Conditional variational autoencoder MM-MARS can assign more granular cell types by meta learning 

![alt text](https://github.com/estelleyao0530/Deep-Learning/blob/main/Figure/mars_result.png)
