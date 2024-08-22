import pandas as pd
import numpy as np
import plotly.express as px

# Function to display histograms for each cluster
def define_cluster_histograms(df, target_column, cluster_column, bin_tags):
    
    clusters = []
    
    for cluster_id in df[cluster_column].unique():
        
        cluster_data = df[df[cluster_column] == cluster_id]
        #display(cluster_data)
        
        if len(list(dict.fromkeys(cluster_data[target_column])))>=len(bin_tags):

            # Calculate histogram center and standard deviation
            hist_center = cluster_data[target_column].mean()
            std_dev = cluster_data[target_column].std()
            
            # Define the bin edges based on standard deviation
            bin_edges = np.linspace(hist_center - (2*std_dev), hist_center + (2*std_dev), len(bin_tags)-1)

#             # Extract the number of bins from the bin_edges list
#             num_bins = int(round(len(cluster_data) ** 0.5))

#             # Create histogram using Plotly Express
#             fig = px.histogram(cluster_data, x=target_column, nbins=num_bins, title='Histogram with {} Equal Sliced Areas'.format(len(bin_tags)))

#             # Add vertical lines for bin edges
#             for edge in bin_edges:
#                 fig.add_shape(
#                     type='line',
#                     x0=edge,
#                     x1=edge,
#                     y0=0,
#                     y1=100,
#                     line=dict(color='red', width=2)
#                 )

#             fig.show()

            # Categorize values into bins
            # 1a Bin é a mais extrema em valores superiores, a última Bin é a mais extrema em valores inferiores
            cluster_data['bin_category'] = pd.cut(cluster_data[target_column], bins=[-np.inf] + list(bin_edges) + [np.inf], labels=bin_tags)

        else:
            print('Could not categorize cluster {}'.format(cluster_id))
        
        clusters.append(cluster_data)            
    
    return pd.concat(clusters)

# --------------------------------------------------
# Definindo as tags

price_tags = ['Super Preço','Bom Preço','Preço Justo','Preço Elevado','Caro']
target_column = 'sre_value'

priced_clustered_base_v_toML = define_cluster_histograms(clustered_base_v_toML, target_column, 'cluster', price_tags)

# Display the resulting dataframe
print(priced_clustered_base_v_toML)