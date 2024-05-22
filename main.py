import numpy as np
import logging
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt

# Load experimental data (dummy data for illustration)
def load_data():
    test_types = ['UT', 'PS', 'EB']  # Modify according to available data
    # test_types.append()
    try:
        data_ut = np.loadtxt('ut_data.dat', delimiter=',')  # Load uniaxial tension data
       
    except FileNotFoundError:
        logging.log(logging.INFO, "No uniaxial tension data found. Using dummy data.")
        data_ut = np.array([])
    try:
        data_ps = np.loadtxt('ps_data.dat', delimiter=',')  # Load pure shear data
        # test_types.append('PS')
    except FileNotFoundError:
        logging.log(logging.INFO, "No pure shear data found. Using dummy data.")
        data_ps = np.array([])

    try:
        data_eb = np.loadtxt('eb_data.dat', delimiter=',')  # Load equibiaxial tension data
        # test_types.append('EB')
        
    except FileNotFoundError:
        logging.log(logging.INFO, "No equilibiaxial data found. Using dummy data.")
        data_eb = np.array([])
       
    
    return data_ut, data_ps, data_eb, test_types

# Define strain energy functions and stress-strain relationships for different models
def strain_energy_ogden(lambdas, mu, alpha):
    return sum(mu_p / alpha_p * (lambdas[0]**alpha_p + lambdas[1]**alpha_p + lambdas[2]**alpha_p - 3) for mu_p, alpha_p in zip(mu, alpha))

# Define stress-strain relationships
def stress_ut_ogden(lambdas, mu, alpha):
    """
    Calculates the stress based on the Ogden model for uniaxial tension.

    Parameters:
    - lambdas: list of strain values
    - mu: list of material parameters mu
    - alpha: list of material parameters alpha

    Returns:
    - Calculated stress value
    """
    lambdax = lambdas[0]
    # return sum(mu_p * (lambdas[0]**(alpha_p - 1) - lambdas[0]**(-(1 + alpha_p / 2))) for mu_p, alpha_p in zip(mu, alpha))
    stresses = np.zeros_like(lambdax)
    for mu_p, alpha_p in zip(mu, alpha):
        term = mu_p * (np.power(lambdax, alpha_p-1) - np.power(lambdax, -(1 + alpha_p / 2)))
        stresses += term

    return stresses


def stress_ps_ogden(lambdas, mu, alpha):
    """
    Calculates the stress based on the Ogden model for pure shear.

    Parameters:
    - lambdas: list of strain values
    - mu: list of material parameters mu
    - alpha: list of material parameters alpha

    Returns:
    - Calculated stress value
    """
    lambdax = lambdas[0]
    stresses = np.zeros_like(lambdax)
    for mu_p, alpha_p in zip(mu, alpha):
        term = mu_p * (np.power(lambdax, alpha_p-1) - np.power(lambdax, -(alpha_p+1)))
        stresses += term

    return stresses

def stress_eb_ogden(lambdas, mu, alpha):
    """
    Calculate the stress for equibiaxial tension using the Ogden model.

    Parameters:
    lambda_x (float): The stretch ratio in the x-direction.
    mu (list of float): The material constants mu for the Ogden model.
    alpha (list of float): The material constants alpha for the Ogden model.

    Returns:
    float: The stress for equibiaxial tension.
    """
    # return sum(mu_k * (lambdas[0]**(alpha_k - 1) - lambdas[0]**(-2*alpha_k - 1)) for mu_k, alpha_k in zip(mu, alpha))
    
    lambdax = lambdas[0]
    stresses = np.zeros_like(lambdax)
        # lambdax = lambdas[0]
    for mu_p, alpha_p in zip(mu, alpha):
        term = mu_p * (np.power(lambdax, alpha_p-1) - np.power(lambdax, -( 2 * alpha_p+1)))
        stresses += term

    return stresses

# Define the objective function for fitting
def objective(params, test_data, test_types):

    # stresses = np.zeros_like(lambdax)
    # for mu_p, alpha_p in zip(mu, alpha):
    #     term = np.power(lambdax, alpha_p-1) - np.power(lambdax, -(1 + alpha_p / 2))
    #     stresses += term

    # return stresses
    mu = params[:len(params)//2]
    alpha = params[len(params)//2:] 
    model_stresses = np.array([], dtype = float)
    test_stresses = np.array([], dtype = float)
    error = 0
    for test_type, data in zip(test_types, test_data):
        if data.shape[0] == 0:
            continue
        lambdas, stresses = data[1:, 0] + 1, data[1:, 1]
        if test_type == 'UT':
            stress_ut = stress_ut_ogden([lambdas, lambdas**(-0.5), lambdas**(-0.5)], mu, alpha) 
            # model_stresses = np.hstack([model_stresses, stress_ut]) 
            ut_error = np.sum((1 - stress_ut / stresses) ** 2)
            # model_stresses = np.hstack([model_stresses, stress_ut])
            # test_stresses = np.hstack([test_stresses, stresses])
            error += ut_error 
        elif test_type == 'PS':
            stress_ps = stress_ps_ogden([lambdas, 1.0, lambdas**(-1)], mu, alpha)
            ps_error = np.sum((1 - stress_ps / stresses) ** 2)
            # test_stresses = np.hstack([test_stresses, stresses])
            # model_stresses = np.hstack([model_stresses, stress_ps])
            error += ps_error 
            # model_stresses = np.hstack([model_stresses, np.array([stress_ps_ogden([l, 1.0, l**(-1)], mu, alpha) for l in lambdas])])
        elif test_type == 'EB':
            # Implement stress relationship for equibiaxial tension
            stress_eb = stress_eb_ogden([lambdas, lambdas, lambdas ** (-2)] , mu, alpha)
            eb_error = np.sum((1 - stress_eb / stresses) ** 2) 
            error += eb_error 
            # test_stresses = np.hstack([test_stresses, stresses])
            # model_stresses = np.hstack([model_stresses, stress_eb])

            # model_stresses = np.hstack([model_stresses, np.array([stress_eb_ogden(l, l, l ** (-2) , mu, alpha) for l in lambdas])])
        # error += np.sum((1 - model_stresses / stresses) ** 2) / len(model_stresses)
    return error

# Perform curve fitting
def fit_hyperelastic_model(test_data, test_types, order):
    """
    Fit a hyperelastic model to experimental data.

    Parameters:
    - test_data: list of arrays containing experimental data for different tests
    - test_types: list of strings specifying the types of tests performed
    - order: integer, order for the hyperelastic model (default is 2)

    Returns:
    - array of optimized parameters for the hyperelastic model
    """
    initial_guess = np.ones(order*2)  #np.random.rand(order * 2)  # Initial guess for mu and alpha (2 parameters each)
    # result = least_squares(objective, initial_guess, args=(test_data, test_types))
    bounds = [(-100, 100)] * order * 2
    result = minimize(objective, initial_guess, args=(test_data, test_types), method='L-BFGS-B', bounds=bounds)
    return result.x

def evaluate_fit(params, test_data, test_types):
    """
    Evaluate the fit of a hyperelastic model based on the given parameters and data.

    Parameters:
    - params: list of model parameters for the hyperelastic model
    - test_data: list of arrays containing experimental data for different test types
    - test_types: list of strings specifying the types of tests performed

    Returns:
    - None
    """
    mu = params[:len(params)//2]
    alpha = params[len(params)//2:]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs = axs.flatten()  # Flatten the array of axes for easy iteration

    for ax, test_type, data in zip(axs, test_types, test_data):
        if data.shape[0] == 0 :
            continue
        lambdas, stresses = data[:, 0] + 1, data[:, 1]
        if test_type == 'UT':
            model_stresses = stress_ut_ogden([lambdas, lambdas**(-0.5), lambdas**(-0.5)], mu, alpha)
            ax.plot(lambdas, stresses, 'o', label=f'{test_type} Data')
            ax.plot(lambdas, model_stresses, '-', label=f'{test_type} Fit')
        elif test_type == 'PS':
            model_stresses = stress_ps_ogden([lambdas, 0, lambdas ** (-1)], mu, alpha) 
            ax.plot(lambdas, stresses, 'o', label=f'{test_type} Data')
            ax.plot(lambdas, model_stresses, '-', label=f'{test_type} Fit')
        elif test_type == 'EB':
            model_stresses = stress_eb_ogden([lambdas, lambdas, lambdas ** (-2)], mu, alpha)
            ax.plot(lambdas, stresses, 'o', label=f'{test_type} Data')
            ax.plot(lambdas, model_stresses, '-', label=f'{test_type} Fit')
        
        ax.set_xlabel('Stretch')
        ax.set_ylabel('Stress')
        ax.legend()
        ax.set_title(f'{test_type} Test')

    plt.tight_layout()
    plt.show()

# Main function to run the fitting process
def main():
    data_ut, data_ps, data_eb, test_types = load_data()
    test_data = [data_ut, data_ps, data_eb]  # Modify according to available data
    
    params = fit_hyperelastic_model(test_data, test_types, order=5)
    evaluate_fit(params, test_data, test_types)

if __name__ == "__main__":
    main()