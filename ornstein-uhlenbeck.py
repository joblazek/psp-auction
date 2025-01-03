import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # NetworkX for small-world construction

# Ornstein-Uhlenbeck process parameters
theta_values = [0.1, 0.2, 0.15]     # rate of mean reversion
sigma_r_values = [0.2, 0.3]         # volatility (for OU process)
sigma_noise_values = [0.05, 0.1]    # white noise standard deviation
dt = 0.01        # time step
num_steps = 1000  # number of time steps

# Small-world network creation
def create_small_world(num_buyers, num_sellers, prob=0.3):
    """Creates a small-world network where buyers bid on a subset of sellers."""
    G = nx.watts_strogatz_graph(num_buyers, num_sellers // 2, prob)
    adjacency_matrix = nx.to_numpy_array(G)
    return adjacency_matrix

# Function to update the small-world network when buyers or sellers are added
def update_network(network, num_buyers, num_sellers):
    """Expands the network dynamically when new buyers or sellers are added."""
    new_network = np.zeros((num_buyers, num_sellers))
    min_rows = min(num_buyers, network.shape[0])
    min_cols = min(num_sellers, network.shape[1])
    
    # Copy the existing connections to the new network
    new_network[:min_rows, :min_cols] = network[:min_rows, :min_cols]
    
    # Create new connections for the added buyers or sellers
    for buyer in range(min_rows, num_buyers):
        for seller in range(min_cols, num_sellers):
            new_network[buyer, seller] = np.random.binomial(1, 0.3)
    
    return new_network

# Ornstein-Uhlenbeck process for adjusting supply-demand ratio
def ou_process_ratio(current_ratio, theta, mu, sigma, dt):
    """Ornstein-Uhlenbeck process to model mean-reverting dynamics of the ratio of buyers to sellers."""
    return current_ratio + theta * (mu - current_ratio) * dt + sigma * np.sqrt(dt) * np.random.randn()

# Function to simulate the auction process with seller price evolution based on buyer bids and noise
def simulate_market_with_evolving_prices(num_steps, theta, mu_ratio, sigma_r, sigma_noise, dt, initial_prices, buyer_bids, buyer_memory, 
                               initial_num_sellers, initial_num_buyers, equilibrium_ratio=1.0):
    prices_over_time = np.zeros((initial_num_sellers, num_steps))
    prices_over_time[:, 0] = initial_prices
    seller_labels = [f"Seller {i+1}" for i in range(initial_num_sellers)]

    # Initial number of buyers and sellers
    current_num_buyers = initial_num_buyers
    current_num_sellers = initial_num_sellers

    # Current ratio of buyers to sellers
    current_ratio = current_num_buyers / current_num_sellers

    # Create the small-world network
    network = create_small_world(current_num_buyers, current_num_sellers)

    for t in range(1, num_steps):
        # Adjust seller prices based on connected buyer bids and white noise
        for seller in range(current_num_sellers):
            # Get the bids from connected buyers
            connected_buyer_bids = buyer_bids[network[:, seller] == 1, seller]
            
            # If there are any connected buyers, adjust seller price based on the average bid
            if len(connected_buyer_bids) > 0:
                average_bid = np.mean(connected_buyer_bids)
            else:
                average_bid = 0  # No connected buyers, no adjustment from bids
            
            # Adjust price based on bids and white noise
            prices_over_time[seller, t] = prices_over_time[seller, t-1] + (average_bid - prices_over_time[seller, t-1]) * 0.5 + np.random.normal(0, sigma_noise)

        # Buyers adjust their bids adaptively with memory of previous adjustments
        for buyer in range(current_num_buyers):
            for seller in range(current_num_sellers):
                if network[buyer, seller] == 1:
                    noise = np.random.normal(0, sigma_noise)
                    buyer_bids[buyer, seller] = max(0, buyer_bids[buyer, seller] + buyer_memory[buyer, seller] + noise)
                    buyer_memory[buyer, seller] = 0.5 * (buyer_bids[buyer, seller] - prices_over_time[seller, t-1])

        # For each seller, determine the second price (second-highest bid)
        for seller in range(current_num_sellers):
            bids_for_seller = buyer_bids[:, seller]
            sorted_bids = np.sort(bids_for_seller)[::-1]  # Sort bids in descending order
            if len(sorted_bids) > 1:
                # Second price auction: winning price is the second-highest bid
                prices_over_time[seller, t] = sorted_bids[1]

    return prices_over_time, seller_labels, current_num_buyers, current_num_sellers

# Running the simulation once with evolving seller prices based on connected buyer bids and noise
theta = np.random.choice(theta_values)
mu_ratio = np.random.uniform(0.8, 1.2)  # Mean ratio of buyers to sellers around equilibrium
sigma_r = np.random.choice(sigma_r_values)
sigma_noise = np.random.choice(sigma_noise_values)
initial_num_sellers = np.random.choice([5, 6, 7])
initial_num_buyers = np.random.randint(10, 15)  # Initial number of buyers is slightly random

# Initial prices and bids
initial_prices = np.random.uniform(0.4, 0.6, initial_num_sellers)
buyer_bids = np.random.uniform(0.3, 0.6, (initial_num_buyers, initial_num_sellers))
buyer_memory = np.zeros((initial_num_buyers, initial_num_sellers))  # Flash memory to remember past bid adjustments

# Set the equilibrium ratio
equilibrium_ratio = 1.0  # Balance point where number of buyers and sellers should converge

# Run the simulation
prices_over_time, seller_labels, final_num_buyers, final_num_sellers = simulate_market_with_evolving_prices(
    num_steps, theta, mu_ratio, sigma_r, sigma_noise, dt, initial_prices, buyer_bids, buyer_memory, 
    initial_num_sellers, initial_num_buyers, equilibrium_ratio
)

# Plotting the output
plt.figure(figsize=(10, 6))
for seller in range(prices_over_time.shape[0]):
    plt.plot(prices_over_time[seller, :], label=seller_labels[seller])

plt.title(f'Evolution of Seller Prices Over Time\nFinal Buyers: {final_num_buyers}, Final Sellers: {final_num_sellers}')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
