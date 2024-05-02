import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import gstools as gs
from sklearn.metrics import mean_squared_error

# This file stores the various custom-defined functions 
def load_data(file_name, protein):        
    # Specify the file path
    if file_name == "tonsil":
        file_path = "/Users/cui/Library/CloudStorage/OneDrive-YaleUniversity/0 High-Dim Spatial/Human Tonsil/tonsil_codex.csv"
    if file_name == "hubmap":
        file_path = "/Users/cui/Library/CloudStorage/OneDrive-YaleUniversity/0 High-Dim Spatial/hubmap/ann/B009A_22_03_03_Skywalker_reg001_compensated_ann.csv"
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    df = df.head(20000)
    if file_name == "tonsil":
        df.rename(columns={'centroid_x': 'x'}, inplace=True)
        df.rename(columns={'centroid_y': 'y'}, inplace=True)
    
    # shuffle the rows
    df = df.sample(frac=1).reset_index(drop=True)
    
    # we normalise x and y so that the grid is approximately 1 by 1
    df['x'] = df['x']/df['x'].max()
    df['y'] = df['y']/df['y'].max()

    # we normalise the protein value and do a standard log transform for potential long-tail effects.
    df[protein] = df[protein] / df[protein].median()
    df[protein] = np.log(1+df[protein])
    #Pick only the large-value outliers, but we don't want to work with those that only give the value = 1.0

    top_20000_rows = df.nlargest(20000, protein)
    top_20000_rows = top_20000_rows.sample(frac=1).reset_index(drop=True)
    df_train = top_20000_rows.head(2000)
    df_test = top_20000_rows.tail(2000)
    
    x_train = df_train['x']
    y_train = df_train['y']
    val_train = df_train[protein]
    
    x_test = df_test['x']
    y_test = df_test['y']
    val_test = df_test[protein]

    return (x_train, y_train, val_train, x_test, y_test, val_test)
    
# The function approximates the gradient
def finite_difference_gradient(params, x, y, val, model, loss_function, model_type, h=1e-5):
    grad = np.zeros_like(params)
    f_orig = loss_function(x,y, val, model, params,model_type)
    for i in range(len(params)):
        # Slightly perturb parameter i
        params_perturbed = np.copy(params)
        params_perturbed[i] += h
        f_perturbed = loss_function(x,y, val, model, params_perturbed,model_type)
        # Compute the gradient approximation
        grad[i] = (f_perturbed - f_orig) / h
    return grad


# The features are the locations (x,y), and labels are val.
def stochastic_gradient_descent(x, y, val, model, loss_function, params, model_type, learning_rate=0.001, epochs=200, batch_size=200, tolerance=1e-8):
    num_samples = len(x)
    previous_loss = float('inf')  # Initialize previous loss to infinity
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        epoch_losses = []
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_x = x[batch_indices]
            batch_y = y[batch_indices]
            batch_val = val[batch_indices]
            
            grad = finite_difference_gradient(params, batch_x, batch_y, batch_val, model, loss_function, model_type)
            params -= learning_rate * grad
            if params[2]<0:
                # params[2]=np.abs(params[2])
                params[2]=0
            # if params[1]<0:
                # params[1]=np.abs(params[1])
            if params[0]<0:
                # params[0]=np.abs(params[0])
                params[0]=0
            
            batch_loss = loss_function(batch_x, batch_y, batch_val, model, params, model_type)
            epoch_losses.append(batch_loss)
        
        average_loss = np.mean(epoch_losses)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {average_loss:.6f}')

    # Check if the loss is no longer decreasing
        if np.abs(previous_loss - average_loss) < tolerance:
            print("Stopping early: Loss no longer decreasing significantly.")
            break
        
        previous_loss = average_loss  # Update previous loss
    return params


# The model for fitting
def model(x,y, val, params, model_type):
    # Define a dictionary mapping model types to their corresponding functions
    model_dict = {
        'Gaussian': gs.Gaussian,
        'Exponential': gs.Exponential,
        'Matern': gs.Matern,
        'Stable': gs.Stable,
        'Rational': gs.Rational,
        'Linear': gs.Linear,
        'Circular': gs.Circular,
        'Spherical': gs.Spherical
    }
    
    # Ensure the model type is valid
    if model_type not in model_dict:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Get the covariance model class from the dictionary
    CovModel = model_dict[model_type]
    
    # Create the covariance model instance
    cov_model = CovModel(dim=2, len_scale=params[0], var=params[1], nugget=params[2],
                         anis=params[3] if 'anis' in CovModel.__init__.__code__.co_varnames else None,
                         angles=params[4] if 'angles' in CovModel.__init__.__code__.co_varnames else None)

    # Initialize the kriging estimator
    EDK = gs.krige.Simple(model=cov_model, cond_pos=(x, y), cond_val=val)
    predictions, sigma = EDK([x, y])
    return predictions

#  trains the model and computes MSE Normalised
def loss_function(x,y, val, model, params,model_type):
    predictions = model(x,y,val, params, model_type)
    return mean_squared_error(predictions, val) / np.var(val)


# The function plots the results
def plotting(params, model_type):
        # Define a dictionary mapping model types to their corresponding functions
    model_dict = {
        'Gaussian': gs.Gaussian,
        'Exponential': gs.Exponential,
        'Matern': gs.Matern,
        'Stable': gs.Stable,
        'Rational': gs.Rational,
        'Circular': gs.Circular,
        'Spherical': gs.Spherical
    }
    
    # Ensure the model type is valid
    if model_type not in model_dict:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"{model_type} plots")
    
    # Get the covariance model class from the dictionary
    CovModel = model_dict[model_type]
    
    # Create the covariance model instance
    cov_model = CovModel(dim=2, len_scale=params[0], var=params[1], nugget=params[2],
                         anis=params[3] if 'anis' in CovModel.__init__.__code__.co_varnames else None,
                         angles=params[4] if 'angles' in CovModel.__init__.__code__.co_varnames else None)
    
    EDK = gs.krige.Simple(
        model=cov_model, 
        cond_pos=(x_train, y_train), 
        cond_val=val_train,
    )
    
    # Second, it displays the training and testing data sets, and the prediction of the model
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))  # 1 row, 3 columns
    axs[0].scatter(x_train, y_train, s=50*val_train, alpha=0.5)
    axs[0].set_title ('Training Data')
    axs[1].scatter(x_test, y_test, s=50*val_test, alpha=0.5)
    axs[1].set_title('Testing Data')
    EDK.structured([gridx, gridy])
    EDK.plot(ax=axs[2])
    axs[2].set_title('Prediction')
    plt.show()
    
    
    # Third, it computes the normalised MSE on train set, demonstrate the errors;
    
    train_val_pred,sigma = EDK([x_train,y_train])
    train_normalized_mse = mean_squared_error(val_train, train_val_pred)/ np.var(val_train)
    print("train_normalized_mse = ", train_normalized_mse)
    
    # print(mean_squared_error(val_pred, val_train) / np.var(val_train))
    
    error = np.abs(val_train - train_val_pred)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
    axs[0].scatter(x_train, y_train, s=15*error,c='r', alpha= 0.5)
    axs[0].set_title ('Absolute Train Errors with Varying Marker Size')
    axs[1].scatter(x_train, y_train, s=error/np.abs(val_train),c='r', alpha= 0.5)
    axs[1].set_title('Normalised Train Errors with Varying Marker Size')
    plt.grid(True)
    plt.show()
    
    
    # Last, it computes the normalised MSE on test set, demonstrate the errors;
    test_val_pred,sigma = EDK([x_test,y_test])
    test_normalized_mse = mean_squared_error(val_test, test_val_pred)/ np.var(val_test)
    print("test_normalised_mse = ", test_normalized_mse)
    
    error = np.abs(val_test - test_val_pred)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
    axs[0].scatter(x_test, y_test, s=15*error,c='r', alpha= 0.5)
    axs[0].set_title ('Absolute Test Errors with Varying Marker Size')
    axs[1].scatter(x_test, y_test, s=error/np.abs(val_test),c='r', alpha= 0.5)
    axs[1].set_title('Normalised Test Errors with Varying Marker Size')
    plt.grid(True)
    plt.show()
    return (train_normalized_mse,test_normalized_mse)




# The function computes train and test errors
def error_calculation(params, model_type):
        # Define a dictionary mapping model types to their corresponding functions
    model_dict = {
        'Gaussian': gs.Gaussian,
        'Exponential': gs.Exponential,
        'Matern': gs.Matern,
        'Stable': gs.Stable,
        'Rational': gs.Rational,
        'Circular': gs.Circular,
        'Spherical': gs.Spherical
    }
    
    # Ensure the model type is valid
    if model_type not in model_dict:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"{model_type} with params {params}")
    
    # Get the covariance model class from the dictionary
    CovModel = model_dict[model_type]
    
    # Create the covariance model instance
    cov_model = CovModel(dim=2, len_scale=params[0], var=params[1], nugget=params[2],
                         anis=params[3] if 'anis' in CovModel.__init__.__code__.co_varnames else None,
                         angles=params[4] if 'angles' in CovModel.__init__.__code__.co_varnames else None)
    
    EDK = gs.krige.Simple(
        model=cov_model, 
        cond_pos=(x_train, y_train), 
        cond_val=val_train,
    )
    
    train_val_pred,sigma = EDK([x_train,y_train])
    train_normalized_mse = mean_squared_error(val_train, train_val_pred)/ np.var(val_train)
    
    test_val_pred,sigma = EDK([x_test,y_test])
    test_normalized_mse = mean_squared_error(val_test, test_val_pred)/ np.var(val_test)
    
    return (train_normalized_mse,test_normalized_mse)
