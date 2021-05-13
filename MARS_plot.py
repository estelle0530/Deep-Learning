def MARS_loss_tracker (training_history, save):
    
    loss = pd.DataFrame.from_dict(training_history['Loss_tracker'])
    epochs = loss.shape[0]
    
    fig, ax = plt.subplots(1,2, figsize=(11, 6.5)) 
    
    ax[0].plot(np.arange(epochs), loss['Test_anno'].values, label = "Unannotated landmark loss")
    ax[0].plot(np.arange(epochs), loss['Train_latent_loss'].values, label = "Train latent loss")
    ax[0].plot(np.arange(epochs), loss['Test_latent_loss'].values, label = "Test latent loss")
    ax[0].set_title("MARS original loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    

    ax[1].plot(np.arange(epochs), loss['Train_reconstr_loss'].values, label = "Train reconstruction loss")
    ax[1].plot(np.arange(epochs), loss['Test_reconstr_loss'].values, label = "Test reconstruction loss")
    ax[1].set_title("Reconstruction loss")
    ax[1].set_xlabel("Epochs")
    ax[1].legend()
    
    
    fig.suptitle("MARS loss across epochs",fontsize=16,y=1)
    plt.tight_layout()
    plt.show()
    
    if save == True:
        fig.savefig("MARS_loss_tracker.png")


def MARS_history(history_train, save):
    
    ###Loss function trajectory
    loss = []
    for l in history_train['Loss']:
        loss.append(l.detach().numpy())
    
    accuracy = history_train['Accuracy']
    
    ##Line plot visuzlie loss per epoch 
    fig, ax = plt.subplots(1,2, figsize=(11, 6.5)) 
    ax[0].scatter(np.arange(len(loss)),loss)
    ax[0].plot(np.arange(len(loss)),loss)
    ax[0].set_xlabel("Epochs", fontsize=12)
    ax[0].set_ylabel("Training loss", fontsize=12)
    ax[0].set_title("Training loss", fontsize=14)
    
    ax[1].scatter(np.arange(len(accuracy)),accuracy)
    ax[1].plot(np.arange(len(accuracy)),accuracy)
    ax[1].set_xlabel("Epochs", fontsize=12)
    ax[1].set_ylabel("Training accuracy", fontsize=12)
    ax[1].set_title("Training accuracy", fontsize=14)
    
    fig.suptitle("Visualization of MARS metrics across epochs",fontsize=16,y=1)
    plt.tight_layout()
    plt.show()    
    
    if save == True:
        fig.savefig("MARS_history.png")

    
def MARS_latent_pca(latent_tracker , epoch_num):
    ###Latent space 
    
    train_latent = latent_tracker[epoch_num]['Train_latent'][0].detach().numpy()
    train_label = latent_tracker[epoch_num]['Train_label'][0].detach().numpy()
    
    test_latent = latent_tracker[epoch_num]['Test_latent'][0].detach().numpy()
    test_label = latent_tracker[epoch_num]['Test_label'][0].detach().numpy()
    
    ###Validation PCA visualization 
    
    pca = PCA(n_components=2)
    pca.fit(train_latent)
    pca_df=pd.DataFrame(pca.transform(train_latent))
    pca_df.columns=['PC1','PC2']
    pca_df=pca_df.copy()
    pca_df['dbscan']=train_label
    
    plt.subplots(1, figsize=(10,8))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="dbscan", palette="deep")
    plt.title("Training latent space")
    plt.show()  
    
    ###Validation PCA visualization 
    pca = PCA(n_components=2)
    pca.fit(test_latent)
    pca_df=pd.DataFrame(pca.transform(test_latent))
    pca_df.columns=['PC1','PC2']
    pca_df=pca_df.copy()
    pca_df['dbscan']=test_label
    
    plt.subplots(1, figsize=(10,8))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="dbscan", palette="deep")
    plt.title("Testing latent space")
    plt.show()

    
def MARS_latent_umap(adata,save, plot_gene_list = ['MARS_labels','experiment',
                                              'CST3','CCL5','FCGR3A','NKG7','MS4A1','CD79A','CD8A']  ):
    
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='MARS_embedding')
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=plot_gene_list, ncols=2 )

    sub_data = adata[adata.obs['experiment']=='Unannotated',]
    if save == True:
        sc.pl.umap(sub_data, color=plot_gene_list, ncols=2, save = '.png' )
    else: 
        sc.pl.umap(sub_data, color=plot_gene_list, ncols=2 )
        
        
def cell_type_assign(adata, save):

    num2name={1:'Malignant',2:'Endothelial',3:'T cell', 4:'Macrophage',
              5:'B cell',6:'CAF',7:'Dendritic',8:'Plasma B', 9:'NK'}

    adata.obs['label_name']=adata.obs['truth_labels'].map(num2name)  
    untable = adata.obs.loc[adata.obs['experiment']=='Unannotated',:]
    
    groundtruth_sum = pd.crosstab(untable['truth_labels'], untable['MARS_labels']).sum(axis=1)
    plot = pd.crosstab(untable['truth_labels'], untable['MARS_labels']).div(groundtruth_sum/100, axis=0).plot.bar(figsize=(17,5), width=1)
    plot.set_xticklabels(['Malignant','Endothelial', 'T cell', 'Macrophage', 'B cell',  'CAF', 'Dendritic', 'Plasma B', 'NK' ],
                         rotation = 45)
    plot.legend(title="MARS labels")
    plot.set_xlabel("Ground truth labels", fontsize=18)
    plot.set_title("Distribution of MARS labels and Ground Truths", fontsize=24)
    plot.set_ylabel("% Truth labels in MARS labels", fontsize=18)
    plt.show()
    
    
    if save==True:
        plot.get_figure().savefig("cell_type_groundtruth.png")
    
    groundtruth_sum = pd.crosstab(untable['MARS_labels'], untable['label_name']).sum(axis=1)
    plot = pd.crosstab(untable['MARS_labels'], untable['label_name']).div(groundtruth_sum/100, axis=0).plot.bar(figsize=(17,5), width=1)
    plot.legend(title="Ground truth")
    plot.set_xlabel("MARS labels", fontsize=18)
    plot.set_ylabel("% MARS labels in Truth labels", fontsize=18)
    plt.show()
    
    if save==True:
        plot.get_figure().savefig("cell_type_MARS.png")

        
def MARS_silhouette(adata, save):
    
    
    anno_data = adata[adata.obs['experiment'] == "Annotated", :]
    unanno_data = adata[adata.obs['experiment'] == "Unannotated", :]
    
    anno_obs = anno_data.obs
    unanno_obs = unanno_data.obs
    
    ###Silhouette
    train_latent = anno_data.obsm['MARS_embedding']
    train_sil=silhouette_samples(train_latent, anno_obs['truth_labels'].values)
    
    val_latent = unanno_data.obsm['MARS_embedding']
    val_sil=silhouette_samples(val_latent, unanno_obs['truth_labels'].values)
    
    n_clusters=len(Counter(adata.obs['truth_labels'].values).keys())
    
    ###Plot silhouette score for train and test 
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 10)
        
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(train_latent) + (n_clusters + 1) * 10])
    
    ax2.set_xlim([-0.1, 1])
    ax2.set_ylim([0, len(val_latent) + (n_clusters + 1) * 10])
         
    y_lower = 10
    val_y_lower = 10
    
    
    name2num={'Malignant':1, 'Endothelial':2, 'T cell':3, 
          'Macrophage':4, 'B cell':5,'CAF':6, 
          'Dendritic':7 ,'Plasma B':8, 'NK':9}

    cluster_list = list(name2num.keys())
    
    for i in range(n_clusters):
        
        
        #cluster_label = list(Counter(adata.obs['truth_labels']).keys())[i]
        cluster_label = cluster_list[i]
        
        #############################################################
        ###############Training 
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        train_ith_cluster_silhouette_values = train_sil[anno_obs['label_name'] == cluster_label]
        train_ith_cluster_silhouette_values.sort()
        
        size_cluster_i = train_ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        #color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, train_ith_cluster_silhouette_values) 
        #edgecolor=color, facecolor=color

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(0.8, y_lower + 0.5 * size_cluster_i, str(cluster_label))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
        
        #############################################################
        ###############Validation 
        val_ith_cluster_silhouette_values = val_sil[unanno_obs['label_name'] == cluster_label]
        val_ith_cluster_silhouette_values.sort()
        
        val_size_cluster_i = val_ith_cluster_silhouette_values.shape[0]
        val_y_upper = val_y_lower + val_size_cluster_i

        #color = cm.nipy_spectral(float(i) / n_clusters)
        ax2.fill_betweenx(np.arange(val_y_lower, val_y_upper),
                          0, val_ith_cluster_silhouette_values)
        # edgecolor=color, facecolor=color

        # Label the silhouette plots with their cluster numbers at the middle
        ax2.text(0.8, val_y_lower + 0.5 * val_size_cluster_i, str(cluster_label))

        # Compute the new y_lower for next plot
        val_y_lower = val_y_upper + 10  # 10 for the 0 samples
     
    ax1.set_title("Training silhouette plot for known labels")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    
    ax2.set_title("Validation silhouette plot for known labels")
    ax2.set_xlabel("Silhouette coefficient values")
    ax2.set_ylabel("Cluster label")
    
    # The vertical line for average silhouette score of all the values
    train_avg=silhouette_score(train_latent, anno_obs['truth_labels'].values)
    val_avg=silhouette_score(val_latent, unanno_obs['truth_labels'].values)
    
    ax1.axvline(x=train_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    ax2.axvline(x=val_avg, color="red", linestyle="--")
    ax2.set_yticks([])  # Clear the yaxis labels / ticks
    ax2.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    plt.suptitle("Silhouette scores for MARS latent space", fontsize=14, fontweight='bold')
    plt.show()  
    
    if save ==True:
        fig.savefig("MARS_silhouette.png")
        
    return train_sil, val_sil, train_avg, val_avg

def MARS_latent_umap_unanno(adata, save, plot_gene_list = ['MARS_labels','experiment',
                                              'CST3','CCL5','FCGR3A','NKG7','MS4A1','CD79A','CD8A']  ):
    
    sub_data = adata[adata.obs['experiment']=='Unannotated',]
    
    sc.pp.neighbors(sub_data, n_neighbors=30, use_rep='MARS_embedding')
    sc.tl.umap(sub_data)

    if save == True:
        sc.pl.umap(sub_data, color=plot_gene_list, ncols=2, save = '.png' )
    else: 
        sc.pl.umap(sub_data, color=plot_gene_list, ncols=2 )