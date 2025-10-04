import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import pandas as pd
import seaborn as sns

def true_function(x):
    """True regression function"""
    return np.sin(1/(x/3 + 0.1))

def generate_data(n, alpha=2, beta_param=2, sigma=1, random_state=None):
    """Generate data from the specified model"""
    np.random.seed(random_state)
    
    # generate X from beta distribution
    X = beta.rvs(alpha, beta_param, size=n)
    # generate y
    Y = true_function(X) + np.random.normal(0, sigma, n)
    
    return X, Y

def estimate_parameters(X, Y, N):
    """Estimate theta_22 and sigma^2 using N blocks based on quantiles"""
    n = len(X)
    
    # Sort data by X for blocking
    sorted_indices = np.argsort(X)
    X_sorted = X[sorted_indices]
    Y_sorted = Y[sorted_indices]
    
    # Create blocks based on quantiles
    quantiles = np.linspace(0, 1, N + 1)
    block_boundaries = np.quantile(X, quantiles)
    blocks_X = []
    blocks_Y = []
    
    for i in range(N):
        # Find indices for current block
        if i == 0:
            mask = X_sorted <= block_boundaries[i + 1]
        elif i == N - 1:
            mask = X_sorted > block_boundaries[i]
        else:
            mask = (X_sorted > block_boundaries[i]) & (X_sorted <= block_boundaries[i + 1])
        
        blocks_X.append(X_sorted[mask])
        blocks_Y.append(Y_sorted[mask])
    
    theta_22_sum = 0
    sigma2_sum = 0 
    theta_total = 0 # to divide by the right amount of data used
    sigma2_total = 0
    
    # Fit polynomials in each block
    for j in range(N):
        if len(blocks_X[j]) < 5:  # Need at least 5 points for polynomial
            continue
            
        X_block = blocks_X[j].reshape(-1, 1)
        Y_block = blocks_Y[j]
        
        # print(f"X= {X_block}")
        polynomial_object = PolynomialFeatures(degree=4)
        X_transformed = polynomial_object.fit_transform(X_block)
        # print(f"X_transformed= {X_transformed}")
        polynomial_regression = LinearRegression()
        polynomial_regression.fit(X_transformed, Y_block)
        
        # Get coefficients for second derivative calculation
        coefficients = polynomial_regression.coef_
        
        b2 = coefficients[2] 
        b3 = coefficients[3] 
        b4 = coefficients[4]
        
        # Calculate second derivative for each point in the block
        for i, x in enumerate(X_block.flatten()):
            m_j_double_prime = 2*b2 + 6*b3*x + 12*b4*x**2
            theta_22_sum += m_j_double_prime**2
        
            X_to_use = X_transformed[i, :].reshape(1, -1)
            y_pred = polynomial_regression.predict(X_to_use)[0]
            sigma2_sum += (Y_block[i] - y_pred)**2

        theta_total += len(X_block)
        sigma2_total += 1
    
    theta_22_hat = theta_22_sum / theta_total 
    sigma2_hat = sigma2_sum / (n - 5*sigma2_total)
    
    return theta_22_hat, sigma2_hat


def compute_h_AMISE(n, theta_22_hat, sigma2_hat):
    """Compute optimal bandwidth using AMISE formula"""
    support_length = 1  # Beta distribution support [0,1]
    h_AMISE = n**(-1/5) * (35 * sigma2_hat * support_length / theta_22_hat)**(1/5)
    return h_AMISE

def compute_Cp(X, Y, N_max):
    """Compute Mallow's Cp for different block sizes using quantile-based blocking"""
    n = len(X)
    N_values = list(range(1, N_max + 1))
    Cp_values = np.zeros(len(N_values))
    
    # Compute RSS for maximum N
    _, sigma2_max = estimate_parameters(X, Y, N_max)
    RSS_max = sigma2_max * (n - 5*N_max)

    if RSS_max < 10e-10:
        return N_values, [np.inf]
    
    for i, N in enumerate(N_values):
        theta_22_hat, sigma2_hat = estimate_parameters(X, Y, N)
        RSS_N = sigma2_hat * (n - 5*N)
        
        if RSS_max > 0 and (n - 5*N_max) > 0:
            Cp = RSS_N / (RSS_max / (n - 5*N_max)) - (n - 10*N)
        else:
            Cp = np.inf
            
        Cp_values[i] = Cp
    
    return N_values, Cp_values

def find_optimal_N(X, Y):
    """Find optimal block size using Mallow's Cp with quantile-based blocking"""
    n = len(X)
    N_max = max(min(n // 20, 10), 2) 
    
    N_values, Cp_values = compute_Cp(X, Y, N_max)
    valid_indices = [i for i, cp in enumerate(Cp_values) if np.isfinite(cp)]
    
    if valid_indices:
        optimal_N = N_values[np.argmin(Cp_values)]
    else:
        optimal_N = 2  # Default to 2 blocks in hostile situations
    
    return optimal_N

def run_simulation_bandwidth(n_values, beta_params, iterations):
    """Simulation function to study effect of sample size and beta distribution"""
    results = {"N":[], "n":[], "(alpha,beta)":[], "h_AMISE": [], "iteration": []}
    
    # Study effect of sample size n
    print("Studying effect of sample size n and parameters of beta...")

    # study effect of sample size n with optimal N, changing also the betas parameters
    for n in n_values:
        for alpha, beta_val in beta_params:
            for i in range(iterations):
                X, Y = generate_data(n, alpha=alpha, beta_param=beta_val, random_state=i*n)
                optimal_N = find_optimal_N(X, Y)
                # print(f"Optimal N: {optimal_N}")
                theta_22_hat, sigma2_hat = estimate_parameters(X, Y, optimal_N)
                
                # Avoid division by zero
                if theta_22_hat > 0:
                    h_AMISE = compute_h_AMISE(n, theta_22_hat, sigma2_hat)
                else:
                    h_AMISE = np.nan

                results["N"].append(optimal_N)
                results["n"].append(n)
                results["(alpha,beta)"].append((alpha, beta_val))
                results["h_AMISE"].append(h_AMISE)
                results["iteration"].append(i)
    
    df_bandwidth = pd.DataFrame(results)
    
    return df_bandwidth


def run_simulation_blocks(N_values, n_values, beta_params, iterations):
    """ function to study the effect of block size on the bandwidth"""
    results_blocks = {"N": [], "n": [], "(alpha,beta)": [], "h_AMISE": [], "n": [], "theta_hat": [], "sigma_hat": []}
    print("Studying effect of block size N...")

    for n in n_values:
        for N in N_values:
            for alpha, beta_val in beta_params:
                # record_H = np.zeros(iterations)
                for i in range(iterations):
                    X, Y = generate_data(n, alpha=alpha, beta_param=beta_val, random_state=N)
                    theta_22_hat, sigma2_hat = estimate_parameters(X, Y, N)
                    
                    if theta_22_hat > 0:
                        h_AMISE = compute_h_AMISE(n, theta_22_hat, sigma2_hat)
                    else:
                        print("problem")
                        h_AMISE = np.nan
                    # record_H[i] = h_AMISE

                    results_blocks["N"].append(N)
                    results_blocks["n"].append(n)
                    results_blocks["(alpha,beta)"].append((alpha, beta_val))
                    results_blocks["h_AMISE"].append(h_AMISE)
                    results_blocks["theta_hat"].append(theta_22_hat)
                    results_blocks["sigma_hat"].append(sigma2_hat)

    df_results_blocks = pd.DataFrame(results_blocks)
    return df_results_blocks
    
def run_simulation_optimal_N(n_values, beta_params, iterations):
    """function to study the relationship between number of samples and the optimal block number N"""
    results_optimal_N = {"n": [], "(alpha,beta)": [], "optimal_N": [], "iteration": []}

    print("Studying relationship between N and n...")
    for n in n_values:
        for alpha, beta_val in beta_params:
            for i in range(iterations):
                X, Y = generate_data(n, alpha=alpha, beta_param=beta_val, random_state=i*n)
                optimal_N = find_optimal_N(X, Y)
                # print(f"Optimal N = {optimal_N}, for n={n}")
                
                results_optimal_N["n"].append(n)
                results_optimal_N["(alpha,beta)"].append((alpha, beta_val))
                results_optimal_N["optimal_N"].append(optimal_N)
                results_optimal_N["iteration"].append(i)
    
    df_optimal_N = pd.DataFrame(results_optimal_N)
    return df_optimal_N


def plot_results(df_optimal_N, df_bandwidth, df_results_blocks):
    """Plot the simulation results using seaborn - Three separate plots"""
    
    # Create string versions of the beta parameters for plotting
    df_results_str = df_bandwidth.copy()
    df_results_str['beta_str'] = df_results_str['(alpha,beta)'].apply(lambda x: f"({x[0]},{x[1]})")

    df_results_blocks_str = df_results_blocks.copy()
    df_results_blocks_str['beta_str'] = df_results_blocks_str['(alpha,beta)'].apply(lambda x: f"({x[0]},{x[1]})")
    
    df_optimal_N_str = df_optimal_N.copy()
    df_optimal_N_str['beta_str'] = df_optimal_N_str['(alpha,beta)'].apply(lambda x: f"({x[0]},{x[1]})")

    #set theme
    sns.set_theme(style="whitegrid")

    # Plot 1: Effect of sample size n
    plt.figure(figsize=(6.5, 4.5))
    sns.lineplot(data=df_results_str, x='n', y='h_AMISE', 
                 hue='beta_str', marker='o', errorbar="sd")
    plt.xlabel('Sample Size n')
    plt.ylabel('h_AMISE')
    plt.title('Effect of Sample Size on Optimal Bandwidth')
    plt.grid(True, alpha=0.3)
    plt.legend(title='(α,β)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("plot1_sample_size_effect.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: Effect of block size N
    plt.figure(figsize=(6.5, 4.5))
    sns.lineplot(data=df_results_blocks_str, x='N', y='h_AMISE', 
                 hue='beta_str', marker='s', errorbar="sd")
    plt.xlabel('Number of Blocks N')
    plt.ylabel('h_AMISE')
    plt.title('Effect of Number of Blocks on Bandwidth')
    plt.grid(True, alpha=0.3)
    plt.legend(title='(α,β)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("plot2_block_size_effect.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # # h_mise vs blocks
    # g = sns.catplot(
    #     data=df_results_blocks_str,
    #     x='N', y='h_AMISE',
    #     col='n', kind='point',
    #     col_wrap=3, sharey=True,
    #     height=3.5, aspect=1.1,
    #     palette='rocket',  
    #     linewidth=1,
    #     hue='n',
    #     legend=False
    # )
    # g.set_titles("Sample size n = {col_name}")
    # g.set_axis_labels("Number of Blocks N", "Bandwidth h_AMISE")
    # for ax in g.axes.flatten():
    #     ax.tick_params(axis='x', rotation=45)
    #     ax.set_xlabel("N") 
    # plt.subplots_adjust(top=0.88)
    # g.fig.suptitle("Bandwidth vs Number of Blocks N", fontsize=18, fontweight='bold')
    # plt.tight_layout()
    # plt.savefig("test_hamise.png", dpi=300, bbox_inches='tight')
    # plt.show()


    # theta vs blocks
    df_results_blocks_str["log_theta"] = np.log(df_results_blocks_str["theta_hat"])
    g = sns.catplot(
        data=df_results_blocks_str,
        x='N', y='log_theta',
        col='n', kind='point',
        col_wrap=3, sharey=True,
        height=3.5, aspect=1.1,
        palette='dark:blue',
        linewidth=1,
        hue='n',
        errorbar="sd",
        capsize=0.2,
        legend=False
    )
    g.set_titles("Sample size n = {col_name}")
    g.set_axis_labels("Number of Blocks N", "Log-Theta")
    for ax in g.axes.flatten():
        ax.set_xlabel("N") 
    plt.subplots_adjust(top=0.88)
    g.fig.suptitle("Bandwidth vs Log-Theta", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig("N_vs_theta.png", dpi=300, bbox_inches='tight')
    plt.show()

    # sigma vs blocks
    g = sns.catplot(
        data=df_results_blocks_str,
        x='N', y='sigma_hat',
        col='n', kind='point',
        col_wrap=3, sharey=True,
        height=3.5, aspect=1.1,
        palette="dark:green",  
        linewidth=1,
        hue='n',
        errorbar="sd",
        legend=False,
        capsize=0.2 
    )
    g.set_titles("Sample size n = {col_name}")
    g.set_axis_labels("Number of Blocks N", "Sigma")
    for ax in g.axes.flatten():
        ax.set_xlabel("N") 
    plt.subplots_adjust(top=0.88)
    g.fig.suptitle("Bandwidth vs Sigma", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig("N_vs_sigma.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 3: Effect of beta distribution
    g = sns.catplot(
        data=df_results_str,
        x='beta_str', y='h_AMISE',
        col='n', kind='box',
        col_wrap=3, sharey=True,
        height=3.5, aspect=1.1,
        palette='rocket',  
        linewidth=1,
        hue='beta_str',
        legend=False
    )
    g.set_titles("Sample size n = {col_name}")
    g.set_axis_labels("Beta Distribution Parameters (α,β)", "Bandwidth h_AMISE")
    for ax in g.axes.flatten():
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel("Beta (α, β)") 
    plt.subplots_adjust(top=0.88)
    g.fig.suptitle("Bandwidth vs Beta Distribution Shape", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plot3_beta_vs_bandwidth.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 4: Effect of size n on optimal block number N
    g = sns.catplot(
        data=df_optimal_N_str,
        x='n', y='optimal_N',
        col='beta_str', kind='point',
        col_wrap=3, sharey=True,
        height=3.5, aspect=1.5,
        hue='beta_str',
        palette="viridis",
        legend=False,
        errorbar='sd', 
        capsize=0.2 
    )
    g.set_titles("Beta = {col_name}")
    g.set_axis_labels("Sample Size n", "Optimal Number of Blocks N")
    for ax in g.axes.flatten():
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel("Sample Size n") 
    plt.subplots_adjust(top=0.88)
    g.fig.suptitle("Optimal Block N vs Sample Size n", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plot4_block_vs_n.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def save_real_distribution():
    x = np.linspace(0, 1, 2000)
    y = true_function(x)

    plt.figure(figsize=(6.5, 3.5))
    plt.plot(x, y, color="blue")
    plt.title(r"Plot of $f(x)=\sin\!\left(\frac{1}{x/3 + 0.1}\right)$ for $0 \leq x \leq 1$")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("real_y.png")
    plt.show()

def plot_beta_distribution(alpha, beta_param, save_path="beta_distribution.png", figsize=(6.5, 4.5)):

    plt.figure(figsize=figsize)
    x = np.linspace(0, 1, 1000)
    y = beta.pdf(x, alpha, beta_param)
    plt.plot(x, y, 'b-', linewidth=2, label=f'Beta({alpha}, {beta_param})')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title(f'Beta Distribution (α={alpha}, β={beta_param})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example usage:
if __name__ == "__main__":
    n_values = [100, 200, 300, 400, 500, 1000] 
    beta_params = [(1, 1), (2, 2), (2, 5), (5, 2), (0.5, 0.5)]
    iterations = 5
    df_bandwidth = run_simulation_bandwidth(n_values, beta_params, iterations)

    N_values = list(range(2, 5+1)) 
    df_results_blocks = run_simulation_blocks(N_values, n_values, beta_params, iterations)

    n_values = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200, 250, 300, 500, 1000, 2000, 5000]
    beta_params = [(1, 1), (2, 5), (5, 2), (0.5, 0.5), (7,2), (2,7)]
    df_optimal_N = run_simulation_optimal_N(n_values, beta_params, iterations)   

    plot_results(df_optimal_N, df_bandwidth, df_results_blocks)
    save_real_distribution()
    plot_beta_distribution(alpha=2, beta_param=5, save_path="beta_distribution1.png", figsize=(6.5, 4.5))
    plot_beta_distribution(alpha=5, beta_param=2, save_path="beta_distribution2.png", figsize=(6.5, 4.5))
    print("plots saved successfully!")
