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
    

    #  Constraints ESG calculated as weighted average
    constraints = [
        ER @ x >= ER_target,            # Return constraint
        esg_scores @ x >= ESG_target,   # ESG constraint   
        cp.sum(x) == 1,                 # Budget constraint
        x >= 0                          # No short selling
    ]
    '''
    # Constraints ESG calculated as hard cap results to more infeasible optimizations but allows short selling easilly
    constraints = [
    ER @ x >= ER_target,
    cp.sum(x) == 1,
    #x >= 0,   # short selling can be allowed when using hard cap
       # Multiply by boolean mask - sets low ESG stocks to zero weight. 
    cp.multiply((esg_scores < ESG_target), x) == 0
]
    '''
    # Solve the problem

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)    


    # checking if portfolio with such constraints exists and only returning ones that do
    if problem.status == 'optimal':
        return x.value
    else:
        print(f"Optimization failed with status: {problem.status}")
        return None
    
    

# Real Data
def load_real_data(returns_file,cov_file,esg_file):
    
    returns_data = pd.read_csv(returns_file, index_col=0).squeeze()
    covariance_matrix_data = pd.read_csv(cov_file, index_col=0)
    esg_data = pd.read_csv(esg_file, index_col=0).squeeze()
        
    # Verify returns and covariance match
    if not np.array_equal(returns_data.index, covariance_matrix_data.index):
        raise ValueError("Returns and covariance RICs don't match!")
    if not np.array_equal(covariance_matrix_data.index, covariance_matrix_data.columns):
        raise ValueError("Covariance matrix not square!")
    
        # Use returns RICs as master list
    master_rics = returns_data.index.tolist()
    
    # Keep only ESG scores for stocks that exist in returns data
    esg_aligned = esg_data[esg_data.index.isin(master_rics)]
    
    # Align all data to the common RICs
    common_rics = esg_aligned.index.tolist()
    returns_aligned = returns_data.loc[common_rics]
    cov_aligned = covariance_matrix_data.loc[common_rics, common_rics]
    
    print(f" Dropped {len(master_rics) - len(common_rics)} stocks without ESG data")
    print(f" Dropped {len(esg_data) - len(common_rics)} extra ESG scores")
    
    # Convert to arrays
    ER = returns_aligned.values
    Sigma = cov_aligned.values
    ESG = esg_aligned.values
    
    
    return ER,Sigma,ESG


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
    
    portfolio_sizes = []  # Track number of stocks

    #storing values of each optimization with different return targets
    returns = []
    risks = []
    actual_esg_scores = []
    
    print(f"Calculating frontier for ESG ≥ {ESG_target}...")
    #Loop that builds the efficient portfolio for each return tarhet
    for i, Rtarget in enumerate(return_targets):
        weights = optimize_portfolio_cvxpy(ER, Sigma, esg_scores, Rtarget, ESG_target) #optimizing weights for each Return target
        
        if weights is not None:
            #n_stocks = np.sum(weights > 0.001)  # Count stocks with >0.1% weight
            #portfolio_sizes.append(n_stocks)
            portfolio_return = ER @ weights                         # return of portfolio (weighted average)
            portfolio_risk = np.sqrt(weights @ Sigma @ weights)     # SD of portfolio (sq of sum of all variance/covariance terms)
            portfolio_esg = esg_scores @ weights                    #  ESG score of portfolio (Weighted average)
            
            returns.append(portfolio_return)
            risks.append(portfolio_risk)
            actual_esg_scores.append(portfolio_esg)
        
        # Print summary
        #if portfolio_sizes:
        #   print(f"  Portfolio sizes: {min(portfolio_sizes)} to {max(portfolio_sizes)} stocks")
    
    return returns, risks, actual_esg_scores


def plot_multiple_esg_frontiers(ER, Sigma, esg_scores):  
    """Plot efficient frontiers for different ESG targets"""
     
    # Find maximum possible ESG (Highest ESG stock), / nyt ei käytetty mutta muuttaa tähän jos toi percentile tapa ei anna järkeviä tuloksia
    esg_min, esg_max = esg_scores.min(), esg_scores.max()
    
    # Adjust ESG targets based on your actual ESG score range 
    
    #targetit voi muuttaa
    esg_targets = [
        np.percentile(esg_scores, 0),    # No constraint (minimum)
        np.percentile(esg_scores, 50),   
        np.percentile(esg_scores, 65),   # Top 60% (better than worst 40%)
        np.percentile(esg_scores, 80),   
        np.percentile(esg_scores, 95)    
    ]
    
    
    
    colors = ['red', 'orange', 'lightgreen', 'blue', 'darkgreen']
    labels = [
        'No ESG Constraint (All stocks)',
        'Top 50% ESG Stocks',
        'Top 65% ESG Stocks', 
        'Top 80% ESG Stocks',
        'Top 95% ESG Stocks'
    ]
    
    
    # Print the actual ESG thresholds / varmistus mitkä ne rajat on
    print("ESG Percentile Targets:")
    percentiles = [0, 30, 50, 70, 95]
    for p, target in zip(percentiles, esg_targets):
        print(f"  {p}th percentile: ESG ≥ {target:.1f}")
    
    
    ''' Plotting the figure  '''
    plt.figure(figsize=(12, 8))
    number_of_points = 10  ## tätä muuttamalla muuttuu frontierin tarkkuus, mutta isommat arvot viä laskutehoo. 100 pistettä ni tulee ok tulokset
    
    for esg_target, color, label in zip(esg_targets, colors, labels): #loop that calculates the efficient frontier with different ESG constraints
                
        returns, risks, actual_esg_scores = calculate_efficient_frontier(             #actual ESG scores not used here, but can be used to do fun stuff (Example in another py file)
            ER, Sigma, esg_scores, esg_target, n_points=number_of_points
        )
        
        # Check if we got valid results (atleast half of the points to form the Efficient frontier) and plots
        if returns and len(returns) >= number_of_points / 2:
            # Convert to annualized for better interpretation 
            annual_returns = np.array(returns) 
            annual_risks = np.array(risks) 
            
            plt.plot(annual_risks, annual_returns, 'o-', color=color, 
                    linewidth=2, markersize=3, label=label, alpha=0.8)
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



def plot_efficient_surface(ER, Sigma, esg_scores):
    ''' kannattaa ajaa isommalla määrällä esg leveleitä, jotta kuvaajasta tulee sulavampi, varmaa pöytäkone hommia. Nyt vielä karvalakkiversio visuaalisesti muutenkin '''
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    esg_levels = [
            # No constraint (minimum)
        np.percentile(esg_scores, 50),   
        np.percentile(esg_scores, 60),   
        np.percentile(esg_scores, 70),   
        np.percentile(esg_scores, 80),   
        np.percentile(esg_scores, 90),   
        np.percentile(esg_scores, 95)
          
    ]
    
    
    risk_grid, return_grid = [], []
    
    for esg_target in esg_levels:
        returns, risks, _ = calculate_efficient_frontier(ER, Sigma, esg_scores, esg_target)
        risk_grid.append(risks)
        return_grid.append(returns)
    
    # Create surface
    X, Y = np.meshgrid(range(len(risk_grid[0])), esg_levels)
    Z = np.array(return_grid)
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Risk Level')
    ax.set_ylabel('ESG Target')
    ax.set_zlabel('Expected Return')
    plt.title('3D Visuaalinen Karvalakkimalli', fontsize=14)
    plt.colorbar(surf)
    plt.show()
    
if __name__ == "__main__":
    
    ER, Sigma, esg_scores = load_real_data(returns_file="Yearly_Returns.csv", cov_file="Covariance_Matrix.csv",esg_file="ESG_scores.csv")
    
    # Plot efficient frontier
    print(f"\n{'='*50}")
    print("PLOTTING EFFICIENT FRONTIER")
    print(f"{'='*50}")
        
    plot_multiple_esg_frontiers(ER, Sigma, esg_scores)    
    #plot_efficient_surface(ER, Sigma, esg_scores)
