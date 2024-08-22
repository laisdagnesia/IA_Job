# Visualização de modelo

from sklearn.tree import plot_tree
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import matplotlib.pyplot as plt
#import networkx as nx
import time

def visualize_best_model(best_model, data, target_col):
    start_global = time.time()
    
    if 'HalvingGridSearchCV' in str(best_model.__class__):
        # Extract the best estimator from HalvingGridSearchCV
        best_estimator = best_model.best_estimator_
        visualize_estimator(best_estimator, data, target_col)
    else:
        visualize_estimator(best_model, data, target_col)
        
    end_global = time.time()    
    print("\nPlot finalzado em {:.2f} min".format((end_global - start_global)/60))

    
def visualize_estimator(estimator, data, target_col):
    model_type = estimator.__class__.__name__

    if model_type == 'DecisionTreeRegressor':
        # Visualize decision tree and print number of nodes and edges
        plt.figure(figsize=(20, 10))
        plot_tree(estimator, filled=True, feature_names=data.drop(target_col, axis=1).columns)
        plt.show()

        num_nodes = estimator.tree_.node_count
        num_edges = num_nodes - 1
        print(f"Number of nodes: {num_nodes}, Number of edges: {num_edges}")

    elif model_type == 'MLPRegressor':
        # Visualize neural network architecture (simplified for illustration)
        plt.figure(figsize=(10, 6))
        plt.title("Neural Network Architecture")
        plt.imshow([len(data.drop(target_col, axis=1).columns), 10, 1], cmap='viridis', aspect='auto', extent=(0, 1, 0, 1))
        plt.axis('off')
        plt.show()

        num_neurons = sum(layer_size for layer_size in estimator.hidden_layer_sizes)
        print(f"Number of neurons: {num_neurons}")

    elif model_type in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
        # Visualize linear regression coefficients or other relevant information
        if hasattr(estimator, 'coef_'):
            plt.figure(figsize=(10, 6))
            plt.bar(data.drop(target_col, axis=1).columns, estimator.coef_)
            plt.title("Linear Regression Coefficients")
            plt.xlabel("Feature")
            plt.ylabel("Coefficient Value")
            plt.show()

            num_coefficients = len(estimator.coef_)
            print(f"Number of coefficients: {num_coefficients}")

    else:
        print(f"Visualization not implemented for {model_type}")


# Example usage:
# visualize_best_model(best_model, data, target_col)