# Testador de modelos de regressão

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, ARDRegression, PassiveAggressiveRegressor, TheilSenRegressor, SGDRegressor, Lars, LassoLars, LassoLarsCV, LassoLarsIC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import StackingRegressor#, RandomSubspace
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import time

def select_best_regression_model(data, target_col, cv_strategy='default', n_jobs=1, cv=None):
    
    start_global = time.time()
    
    scores = {}
    
    # Regression Models
    regression_models = [
        LinearRegression(),
        Ridge(alpha=1.0),
        Lasso(alpha=1.0),
        ElasticNet(alpha=1.0, l1_ratio=0.5),
        BayesianRidge(),
        HuberRegressor(),
        GaussianProcessRegressor(),
#         RandomSubspace(),  # Ensemble method
        StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor()),
                ('gb', GradientBoostingRegressor())
            ],
            final_estimator=LinearRegression()
        ),
        OrthogonalMatchingPursuit(),
        OrthogonalMatchingPursuitCV(),
        ARDRegression(),
        PassiveAggressiveRegressor(),
        TheilSenRegressor(),
        SVR(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        GradientBoostingRegressor(n_iter_no_change=10, tol=0.01),  # Early stopping
        MLPRegressor(max_iter=1000, early_stopping=True, n_iter_no_change=10),  # Early stopping
        SGDRegressor(),
        Lars(),
        LassoLars(),
        LassoLarsCV(),
        LassoLarsIC()
    ]

    # Create KFold cross-validation object
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for model in regression_models:
        model_name = model.__class__.__name__
        print('Testing model: {}'.format(model_name))

        try:
            if model_name in ['PCA', 'KernelPCA', 'TSNE']:
                model.fit(data.drop(target_col, axis=1))
                transformed_data = model.transform(data.drop(target_col, axis=1))
                reg_model = LinearRegression().fit(transformed_data, data[target_col])
                predictions = reg_model.predict(transformed_data)
            else:
                if model_name in ['Ridge', 'Lasso', 'ElasticNet']:
                    # Regularization for specific models
                    param_grid = {'alpha': [0.1, 1.0, 10.0]}
                    model = HalvingGridSearchCV(model, param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=n_jobs)

                # Hyperparameter Tuning
                if cv_strategy == 'grid_search':
                    param_grid = {}  # Add hyperparameters for tuning
                    model = HalvingGridSearchCV(model, param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=n_jobs)

                if hasattr(model, 'n_iter_no_change') and hasattr(model, 'tol'):
                    # Early Stopping for models that support it
                    model.n_iter_no_change = 10  # Number of consecutive iterations with no improvement
                    model.tol = 0.01  # Tolerance to declare convergence

                model.fit(data.drop(target_col, axis=1), data[target_col])
                predictions = model.predict(data.drop(target_col, axis=1))

            # Regression Model Scoring
            if cv_strategy != 'grid_search':
                # Cross-Validation Strategy
                if cv_strategy == 'stratified_kfold':
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                else:
                    cv_scores = cross_val_score(model, data.drop(target_col, axis=1), data[target_col], cv=kf, scoring='neg_mean_squared_error')
                    cv_score = cv_scores.mean()
            else:
                cv_score = None

            # Regression scores
            r2 = r2_score(data[target_col], predictions)
            mse = mean_squared_error(data[target_col], predictions)
            mae = mean_absolute_error(data[target_col], predictions)

            scores[model_name] = {'trained_model': model, 'r2_score': r2, 'mse_score': mse, 'mae_score': mae, 'cv_score': cv_score}

        except Exception as erro:
            print("Deu ruim: {}".format(erro))

            
    # -----------------------------------------------
    # Selecting the best model based on cost-benefit score
    scores_df = pd.DataFrame(scores).T.reset_index().rename(columns={'index':'model'})
    
    scores_df['final_score'] =  scores_df['mse_score'] / scores_df['r2_score']
    
    positive_scores_df = scores_df[scores_df['r2_score']>0].sort_values(by='final_score', ascending=True).reset_index(drop=True)
    
    best_model = list(positive_scores_df['trained_model'])[0]
            
    # Performance Visualization for the best model
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[target_col], y=best_model.predict(data.drop(target_col, axis=1)))
    plt.title(f"True vs. Predicted Values - {best_model.__class__.__name__}")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.show()

    # Learning Curve for the best model
    plt.figure(figsize=(10, 6))
    plot_learning_curve(best_model, best_model.__class__.__name__, data.drop(target_col, axis=1), data[target_col], cv=cv)
    plt.show()

    # Model Persistence
    # Save the best model to disk for later use
    #joblib.dump(best_model, 'best_model.pkl')

    
    end_global = time.time()
    print('\nThe best Regression model is:', list(positive_scores_df['model'])[0])
    
    print("\nEstudo finalzado em {:.2f} segundos".format((end_global - start_global)))
    
    return best_model, scores_df



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    """
    plt.figure(figsize=(10, 6))
    plt.title(f"Learning Curve - {title}")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# # Example usage:
# target_col = 'sre_value'
# best_model, scores_df = select_best_regression_model(data, target_col, cv_strategy='grid_search', n_jobs=-1, cv=None)
# display(scores_df)
# best_model