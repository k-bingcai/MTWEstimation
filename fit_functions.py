# Functions to fit MTW regression 

# Import libraries 
from mtw.mtw import MTW
import numpy as np
import warnings 

def fit_WassVAR1(ts_data_list,
                 wassPen_vals,
                 ground_M,
                 max_iter = 50000):

    """
    Fits multitask VAR with Wasserstein Penalty

    ts_data: 
        a list of 2D numpy arrays
    """

    # Sanity check 
    assert type(ts_data_list) is list 

    # Get dimensions 
    n_tasks = len(ts_data_list)
    n_samples, n_features = ts_data_list[0].shape

    # Trial mode 
    warnings.warn("TRIAL: We do not use L1 penalty for now.")
    warnings.warn("LIMITATION: the number of samples cannot differ per task for now.")

    # Function to convert TS to regression format 
    def convert_to_regression(data_matrix):

        # For VAR(1) only
        p = 1

        # Get dimensions 
        T, d = data_matrix.shape
        N = T - p

        # Get outcome variables 
        Y = data_matrix[p:]
        Y = Y.T.reshape(-1, 1)

        # Get predictor
        X = data_matrix[0:T-1] 

        # Form kronecker product
        Z = np.kron(np.eye(d), X)

        return Y, Z

    # Initialize containers 
    Xs = [] 
    Ys = []

    # Convert to regression
    for data_k in ts_data_list:
        Y_k, predictors_k = convert_to_regression(data_k)
        Ys.append(Y_k)
        Xs.append(predictors_k)

    # Stack into correct format 
    Xs_array = np.stack(Xs, axis=0)  # shape: (n_tasks, n_samples, n_features)
    Ys_array = np.stack(Ys, axis=0).squeeze(-1)  # shape: (n_tasks, n_samples, 1)

    # Check shapes
    assert Xs_array.shape == (n_tasks, (n_samples - 1) * n_features, n_features ** 2) 
    assert Ys_array.shape == (n_tasks, (n_samples - 1) * n_features)

    # Create container for results 
    out_results = []

    # Loop through wassPen values 
    d = n_features # Convenience
    for wassP_i in wassPen_vals:
        
        print("[INFO] Fitting for wassPen value of {}".format(wassP_i))

        # Define model 
        mtw_model = MTW(alpha = wassP_i,
                        beta = 0,
                        max_iter = max_iter, 
                        M = ground_M)

        # Fit the model 
        mtw_model.fit(Xs_array, Ys_array)

        # Extract matrices 
        est_matrices_i = [beta.reshape(d,d) for beta in mtw_model.coefs_.T]

        # Add to results 
        out_results.append({'wassPen': wassP_i,
                            'est_Phi': est_matrices_i})

    return out_results
