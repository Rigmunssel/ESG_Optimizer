import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

#Muuta lopussa
plt.style.use("bmh")

def optimize_portfolio_cvxpy(ER, Sigma, esg_scores, ER_target, ESG_target):
    """ 
    Optimoi yksittäisen portfolion
    Solve the Mean-Variance-ESG optimization problem using CVXPY
    
    Parameters:
    ER: array (n_assets,) - Expected returns
    Sigma: array (n_assets, n_assets) - Covariance matrix
    esg_scores: array (n_assets,) - ESG scores
    ER_target: Minimum required return (float)
    ESG_target: Minimum required ESG score (float)
    
    Returns:
    weights: array (n_assets,) - Optimal portfolio weights
    """
    # Number of assets
    n_assets = len(ER)
    
    #  Optimization variable (Vector representing portfolio weights for each of len(ER))
    x = cp.Variable(n_assets)
    
    #  Objective function: minimize portfolio variance (cp.quad_form computes portfolio variance)
    objective = cp.Minimize(cp.quad_form(x, Sigma))
    
    #  Constraints
    constraints = [
        ER @ x >= ER_target,            # Return constraint
        esg_scores @ x >= ESG_target,   # ESG constraint
        cp.sum(x) == 1,                 # Budget constraint
        x >= 0                          # No short selling
    ]

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # checking if portfolio with such constraints exists and only returning ones that do
    if problem.status == 'optimal':
        return x.value
    else:
        print(f"Optimization failed with status: {problem.status}")
        return None
    
    
                                               ### fake data ###
def generate_data(n_assets=500):
        
    np.random.seed(42)
    
    # Base parameters (more realistic)
    base_volatility = 0.15 / np.sqrt(252)  # 15% annual vol → daily
    base_return = 0.08 / 252  # 8% annual return → daily
    
    # Expected returns with some dispersion
    ER = np.random.normal(base_return, base_return * 0.5, n_assets)
    ER = np.clip(ER, base_return * 0.3, base_return * 2.0)  # Reasonable bounds
    
    # Generate realistic correlation structure
    # Use a market factor model
    market_betas = np.random.uniform(0.7, 1.3, n_assets)  # Realistic betas
    specific_vol = np.random.uniform(0.1, 0.3, n_assets) * base_volatility
    
    # Create correlation matrix with market structure
    corr_matrix = np.outer(market_betas, market_betas) * 0.3  # Market explains 30% of correlation
    np.fill_diagonal(corr_matrix, 1.0)  # Perfect self-correlation
    
    # Add some random correlation noise
    noise = np.random.uniform(-0.1, 0.1, (n_assets, n_assets))
    noise = (noise + noise.T) / 2  # Make symmetric
    np.fill_diagonal(noise, 0)  # No noise on diagonal
    
    corr_matrix = corr_matrix + noise
    corr_matrix = np.clip(corr_matrix, -0.5, 0.9)  # Realistic correlation bounds
    
    # Ensure positive definite
    eigenvalues = np.linalg.eigvals(corr_matrix)
    min_eigenval = np.min(eigenvalues)
    if min_eigenval < 1e-6:
        corr_matrix += np.eye(n_assets) * (abs(min_eigenval) + 1e-6)
    
    # Create volatility vector
    volatilities = np.random.uniform(0.8, 1.5, n_assets) * base_volatility
    
    # Convert to covariance matrix
    D = np.diag(volatilities)
    Sigma = D @ corr_matrix @ D
    
    # Add idiosyncratic noise to ensure no zero risk
    Sigma += np.eye(n_assets) * (base_volatility * 0.1) ** 2
    
    # ESG scores with realistic distribution
    esg_scores = np.random.beta(2, 2, n_assets) * 40 + 50  # Centered around 70
    
    print(f"Generated data statistics:")
    print(f"  Return range: [{ER.min():.6f}, {ER.max():.6f}]")
    print(f"  Volatility range: [{np.sqrt(np.diag(Sigma)).min():.6f}, {np.sqrt(np.diag(Sigma)).max():.6f}]")
    print(f"  Correlation range: [{np.min(corr_matrix):.3f}, {np.max(corr_matrix):.3f}]")
    print(f"  ESG range: [{esg_scores.min():.1f}, {esg_scores.max():.1f}]")
    
    return ER, Sigma, esg_scores


        ### Ääriarvot ###



def find_achievable_return_range(ER, Sigma, esg_scores, lambda_target):
    """Find what return targets are actually achievable with ESG constraint"""
    
    
    # Find maximum achievable return with ESG & Short selling constraints
    n_assets = len(ER)
    x_max = cp.Variable(n_assets)
    prob_max = cp.Problem(cp.Maximize(ER @ x_max),
                         [cp.sum(x_max) == 1, 
                          x_max >= 0,
                          esg_scores @ x_max >= lambda_target])
    prob_max.solve()
    
    
    if prob_max.status == 'optimal':
        max_achievable_return = ER @ x_max.value
        max_esg_actual = esg_scores @ x_max.value
        print(f"  Maximum return with ESG constraint: {max_achievable_return:.6f} (ESG: {max_esg_actual:.1f})")
    else:
        # If constrained max fails, use unconstrained max (Should not be the case)
        max_achievable_return = np.max(ER)
        print(f" ❌ Using unconstrained max return: {max_achievable_return:.6f}")
    
    # Find minimum return (global minimum variance with ESG constraint)
    x_min = cp.Variable(n_assets)
    prob_min = cp.Problem(cp.Minimize(cp.quad_form(x_min, Sigma)),
                         [cp.sum(x_min) == 1, 
                          x_min >= 0,
                          esg_scores @ x_min >= lambda_target])
    prob_min.solve()
    
    if prob_min.status == 'optimal':
        min_achievable_return = ER @ x_min.value
        min_esg_actual = esg_scores @ x_min.value
        print(f"  Minimum return with ESG constraint: {min_achievable_return:.6f} (ESG: {min_esg_actual:.1f})")
    else:
        # Fallback to unconstrained minimum (Shouldnt be happening)
        x_min_simple = cp.Variable(n_assets)
        prob_min_simple = cp.Problem(cp.Minimize(cp.quad_form(x_min_simple, Sigma)),
                                    [cp.sum(x_min_simple) == 1, x_min_simple >= 0])
        prob_min_simple.solve()
        min_achievable_return = ER @ x_min_simple.value
        print(f" ❌ Using unconstrained min return: {min_achievable_return:.6f}")
    
    return min_achievable_return, max_achievable_return


    
    ###     Efficient Frontiers     ###
    
def calculate_efficient_frontier(ER, Sigma, esg_scores, ESG_target, n_points=15): #n_point changes with how many points the frontier is calculated with
    """Calculate efficient frontier points for one portfolio"""
    
    # First find what's actually achievable
    min_return, max_return = find_achievable_return_range(ER, Sigma, esg_scores, ESG_target)
    
    # From max and min returns create return targets, that will be optimized
    return_targets = np.linspace(min_return, max_return, n_points)
    
    #storing values of each optimization with different return targets
    returns = []
    risks = []
    actual_esg_scores = []
    
    print(f"Calculating frontier for ESG ≥ {ESG_target}...")
    #Loop that builds the efficient portfolio for each return tarhet
    for i, Rtarget in enumerate(return_targets):
        weights = optimize_portfolio_cvxpy(ER, Sigma, esg_scores, Rtarget, ESG_target) #optimizing weights for each Return target
        
        if weights is not None:
            portfolio_return = ER @ weights                         # return of portfolio (weighted average)
            portfolio_risk = np.sqrt(weights @ Sigma @ weights)     # SD of portfolio (sq of sum of all variance/covariance terms)
            portfolio_esg = esg_scores @ weights                    #  ESG score of portfolio (Weighted average)
            
            returns.append(portfolio_return)
            risks.append(portfolio_risk)
            actual_esg_scores.append(portfolio_esg)
    
    return returns, risks, actual_esg_scores


def plot_multiple_esg_frontiers(ER, Sigma, esg_scores):
    """Plot efficient frontiers for different ESG targets"""
     
    # Find maximum possible ESG (Highest ESG stock)
    max_esg = np.max(esg_scores)  # Can be used when defining ESG targets by making targets % of MAX ESG. Have to check with real data if this is useful.

    
    # Define multiple ESG targets 
    esg_targets = [0,50, 60, 65, 70] # Now just random numbers
    colors = ['red', 'orange', 'lightgreen', 'blue', 'darkgreen']
    labels = ['No ESG Constraint', '70% Max ESG', '75% Max ESG', '85% Max ESG', '95% Max ESG']
    
    ''' Plotting the figure  '''
    plt.figure(figsize=(12, 8))
    number_of_points = 15
    
    for esg_target, color, label in zip(esg_targets, colors, labels): #loop that calculates the efficient frontier with different ESG constraints
                
        returns, risks, actual_esg_scores = calculate_efficient_frontier(             #actual ESG scores not used here, but can be used to do fun stuff (Example in another py file)
            ER, Sigma, esg_scores, esg_target, n_points=number_of_points
        )
        
        # Check if we got valid results (atleast half of the points to form the Efficient frontier) and plots
        if returns and len(returns) >= number_of_points / 2:
            # Convert to annualized for better interpretation OIKEELLA DATALLLA TÄYTYY MUUTTAA❌❌❌❌❌
            annual_returns = np.array(returns) * 252 * 100
            annual_risks = np.array(risks) * np.sqrt(252) * 100
            
            plt.plot(annual_risks, annual_returns, 'o-', color=color, 
                    linewidth=2, markersize=5, label=label, alpha=0.8)
            print(f"  ✅ {len(returns)} points")
        else:
            print(f"  ❌ Not enough points for {label}")
    
    #plot feke
    plt.xlabel('Annual Risk (Standard Deviation %)', fontsize=12)
    plt.ylabel('Annual Expected Return (%)', fontsize=12)
    plt.title('Efficient Frontiers for Different ESG Targets', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Testi
if __name__ == "__main__":
    
    print("Generating sample data...")
    ER, Sigma, esg_scores = generate_data(n_assets=500)
    
    print(f"\nData Summary:")
    print(f"  Number of assets: {len(ER)}")
    print(f"  Expected returns - Min: {ER.min():.6f}, Max: {ER.max():.6f}")
    print(f"  ESG scores - Min: {esg_scores.min():.1f}, Max: {esg_scores.max():.1f}")
    print(f"  Covariance matrix condition number: {np.linalg.cond(Sigma):.2f}")
    
    # Test single portfolio first
    print(f"\n{'='*50}")
    print("TESTING SINGLE PORTFOLIO")
    print(f"{'='*50}")
    
    # Use a moderate ESG target that's likely achievable
    test_eta = 0.000007
    test_lambda = 70.0
    
    print(f"Testing with return target: {test_eta:.6f}, ESG target: {test_lambda:.1f}")
    weights = optimize_portfolio_cvxpy(ER, Sigma, esg_scores, test_eta, test_lambda)
    
    if weights is not None:
        print("✅ Single portfolio optimization successful!")
        print(f"   Actual return: {ER @ weights:.6f}")
        print(f"   Actual risk: {np.sqrt(weights @ Sigma @ weights):.6f}")
        print(f"   Actual ESG: {esg_scores @ weights:.1f}")
        print(f"   Assets used: {(weights > 0.001).sum()}/{len(weights)}")
    else:
        print("❌ Single portfolio optimization failed")
        print("   Trying with lower ESG target...")
        weights = optimize_portfolio_cvxpy(ER, Sigma, esg_scores, test_eta, 60.0)
        if weights is not None:
            print("✅ Success with lower ESG target!")

    
    # Plot efficient frontier
    print(f"\n{'='*50}")
    print("PLOTTING EFFICIENT FRONTIER")
    print(f"{'='*50}")
    

    plot_multiple_esg_frontiers(ER, Sigma, esg_scores)
