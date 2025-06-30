import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random


np.random.seed()


# Modified to track convergence
class Node:
    def __init__(self, id, price, quantity, gamma=1.0):
        self.id = id
        self.price = price
        self.quantity = quantity  # Positive for buyers, negative for sellers
        self.bid = np.random.uniform(50, 100)  # Initial bid for buyers
        self.gamma = gamma  # Elasticity parameter for buyers
        self.valuation_function = lambda z: self.gamma * np.log(1 + z)

# Define a convergence threshold
CONVERGENCE_THRESHOLD = 1e-3
MAX_ITERATIONS = 1000  # Set a maximum iteration limit to prevent infinite loops

# Function to calculate the Euclidean norm of the change in bids or prices
def calculate_convergence(bids_or_prices_previous, bids_or_prices_current):
    common_keys = set(bids_or_prices_previous.keys()).intersection(bids_or_prices_current.keys())
    previous_bids_list = [bids_or_prices_previous[node_id] for node_id in sorted(common_keys)]
    current_bids_list = [bids_or_prices_current[node_id] for node_id in sorted(common_keys)]
    return np.linalg.norm(np.array(previous_bids_list) - np.array(current_bids_list))

# Function to track convergence
def has_converged(previous_bids, current_bids):
    return calculate_convergence(previous_bids, current_bids) < CONVERGENCE_THRESHOLD

# Create a random market network and the buyer-buyer network
def create_random_market_network(buyers, sellers, connectivity_prob):
    G_seller_buyer = nx.Graph()

    for seller in sellers:
        G_seller_buyer.add_node(seller.id, obj=seller)
    for buyer in buyers:
        G_seller_buyer.add_node(buyer.id, obj=buyer)

    # Connect buyers to sellers with a given connectivity probability
    for seller in sellers:
        for buyer in buyers:
            if random.random() < connectivity_prob:
                G_seller_buyer.add_edge(seller.id, buyer.id)

    return G_seller_buyer

# Create seller-seller network based on shared buyers
def create_seller_seller_network(buyers, sellers, G_seller_buyer):
    print("Creating seller-seller network based on shared buyers")
    
    # Initialize an empty graph for seller-seller relationships
    G_seller_seller = nx.Graph()

    # Add all sellers as nodes in the seller-seller network
    for seller in sellers:
        G_seller_seller.add_node(seller.id, obj=seller)
    
    # Iterate through all pairs of sellers to find shared buyers
    for i, seller_1 in enumerate(sellers):
        for seller_2 in sellers[i+1:]:
            # Find common buyers between seller_1 and seller_2
            buyers_seller_1 = set(G_seller_buyer.neighbors(seller_1.id))
            buyers_seller_2 = set(G_seller_buyer.neighbors(seller_2.id))
            shared_buyers = buyers_seller_1.intersection(buyers_seller_2)
            
            if shared_buyers:
                # Add an edge between seller_1 and seller_2 if they share buyers
                G_seller_seller.add_edge(seller_1.id, seller_2.id, shared_buyers=list(shared_buyers))

    return G_seller_seller
    
# Create buyer-buyer network based on shared sellers
def create_buyer_buyer_network(buyers, sellers, G_seller_buyer):
    print("Creating buyer-buyer network based on shared sellers")
    
    # Create buyer-buyer connections based on shared sellers
    G_buyer_buyer = nx.Graph()
    for buyer in buyers:
        G_buyer_buyer.add_node(buyer.id, obj=buyer)

    # Connect buyers through shared sellers
    for seller in sellers:
        connected_buyers = [buyer for buyer in G_seller_buyer.neighbors(seller.id)]
        for i in range(len(connected_buyers)):
            for j in range(i + 1, len(connected_buyers)):
                G_buyer_buyer.add_edge(connected_buyers[i], connected_buyers[j])

    return G_buyer_buyer
    
    
# Detect and add edges to isolated nodes
def check_and_fix_isolated_nodes(G_seller_buyer, buyers, sellers):
    # Check for isolated sellers
    isolated_sellers = [seller.id for seller in sellers if len(list(G_seller_buyer.neighbors(seller.id))) == 0]
    isolated_buyers = [buyer.id for buyer in buyers if len(list(G_seller_buyer.neighbors(buyer.id))) == 0]

    if isolated_sellers:
        print(f"Warning: Isolated sellers detected: {isolated_sellers}. Adding random buyer connections.")
        for seller_id in isolated_sellers:
            # Randomly connect the isolated seller to one or more buyers
            random_buyer = random.choice(buyers)
            G_seller_buyer.add_edge(seller_id, random_buyer.id)
            random_buyer = random.choice(buyers)
            G_seller_buyer.add_edge(seller_id, random_buyer.id)
            
    if isolated_buyers:
        print(f"Warning: Isolated buyers detected: {isolated_buyers}. Adding random seller connections.")
        for buyer_id in isolated_buyers:
            # Randomly connect the isolated buyer to one or more sellers
            random_seller = random.choice(sellers)
            G_seller_buyer.add_edge(random_seller.id, buyer_id)

    return G_seller_buyer

# Opt-out function based on utility maximization (simplified for this example)
def opt_out_utility_based(buyer, connected_sellers, G_seller_buyer, gamma=1.0):
    total_demand = buyer.quantity
    chosen_sellers = random.sample(connected_sellers, k=random.randint(1, len(connected_sellers)))
    return chosen_sellers

def update_bids_with_utility_opt_out(G_seller_buyer, G_buyer_buyer, G_seller_seller, buyer_config, seller_config, price_history, buyer_price_history, use_opt_out=True):
    previous_bids = {node_id: G_seller_buyer.nodes[node_id]['obj'].bid for node_id in G_seller_buyer.nodes if "Buyer" in node_id}
    iteration = 0
    converged = False
    
    while not converged and iteration < MAX_ITERATIONS:
        iteration += 1
        current_bids = {}
        
        for node_id in G_seller_buyer.nodes:
            node = G_seller_buyer.nodes[node_id]['obj']

            # Logic for Buyers: Adjust bids based on seller prices and competition
            if "Buyer" in node_id:
                seller_prices = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_buyer.neighbors(node_id) if "Seller" in neighbor]
                if seller_prices:
                    min_seller_price = min(seller_prices)  # Buyers prefer the lowest seller price
                    other_buyer_bids = [G_buyer_buyer.nodes[neighbor]['obj'].bid for neighbor in G_buyer_buyer.neighbors(node_id)]
                    
                    # If there are competing buyers, adjust bids competitively (PSP logic)
                    if other_buyer_bids:
                        second_highest_buyer_bid = sorted(other_buyer_bids)[-2] if len(other_buyer_bids) > 1 else other_buyer_bids[0]
                        node.bid = (min_seller_price + second_highest_buyer_bid) / 2  # PSP influenced bid
            
            # Logic for Sellers: Adjust prices based on bids and seller-seller influence
            elif "Seller" in node_id:
                buyer_bids = [G_seller_buyer.nodes[neighbor]['obj'].bid for neighbor in G_seller_buyer.neighbors(node_id) if "Buyer" in neighbor]
                if len(buyer_bids) > 1:
                    second_highest_bid = sorted(buyer_bids)[-2]  # PSP: Second-highest bid
                    node.price = second_highest_bid  # Sellers price at second-highest buyer bid

                # Seller-Seller influence: Adjust based on prices of connected sellers
                connected_seller_prices = [G_seller_seller.nodes[neighbor]['obj'].price for neighbor in G_seller_seller.neighbors(node_id)]
                if connected_seller_prices:
                    avg_seller_price = np.mean(connected_seller_prices)
                    node.price = (node.price + avg_seller_price) / 2  # Adjust price based on seller neighbors
    
            # Store current bids/prices for history tracking
            current_bids[node_id] = node.bid
            if "Buyer" in node_id:
                buyer_price_history[node_id].append(node.bid)
            if "Seller" in node_id:
                price_history[node_id].append(node.price)

        # Check for convergence using dictionaries
        if has_converged(previous_bids, current_bids):
            converged = True
        
        previous_bids = current_bids.copy()
    
    return iteration  # Return the number of iterations to convergence

# Plot the network and price history
def plot_network_and_price_history(price_history, buyer_price_history, G_seller_buyer, G_buyer_buyer, seller_buyer_connections):
    pos = nx.spring_layout(G_seller_buyer)

    plt.figure(figsize=(12, 10))

    # Basic colors for the 5 sellers in the legend
    basic_colors = ['red', 'blue', 'green', 'purple', 'orange']
    # Full spectrum for actual plot color coding
    full_spectrum_colors = plt.cm.rainbow(np.linspace(0, 1, len(seller_buyer_connections)))

    # First subplot: Combined seller-buyer network with seller clusters
    plt.subplot(2, 1, 1)
    for idx, (seller_id, buyers) in enumerate(seller_buyer_connections.items()):
        color = full_spectrum_colors[idx]  # Use the full spectrum for clusters
        # Draw the seller node in the cluster color
        nx.draw_networkx_nodes(G_seller_buyer, pos, nodelist=[seller_id], node_color=[color], node_size=500, label=f"Seller {seller_id}")
        # Draw the buyer nodes in the same cluster color
        buyer_ids = buyers
        nx.draw_networkx_nodes(G_seller_buyer, pos, nodelist=buyer_ids, node_color=[color] * len(buyer_ids), node_size=500)
        # Draw seller-buyer edges in the cluster color
        for buyer_id in buyer_ids:
            nx.draw_networkx_edges(G_seller_buyer, pos, edgelist=[(seller_id, buyer_id)], edge_color=[color], style='solid', width=2)

    # Draw the buyer-buyer network (dotted edges)
    dotted_edges = [(u, v) for u, v in G_buyer_buyer.edges]
    nx.draw_networkx_edges(G_buyer_buyer, pos, edgelist=dotted_edges, style='dotted', edge_color='blue')

    # Add labels and title
    nx.draw_networkx_labels(G_seller_buyer, pos)
    plt.title("Buyer-Seller Network with Buyer-Buyer Connections")

    # Second subplot: Plot seller price adjustments and buyer price adjustments over time
    plt.subplot(2, 1, 2)
    for idx, (seller_id, buyers) in enumerate(seller_buyer_connections.items()):
        color = full_spectrum_colors[idx]  # Use full spectrum for plot
        # Plot seller's price as a solid line
        plt.plot(np.arange(len(price_history[seller_id])), price_history[seller_id], linestyle='solid', color=color, label=f"Seller {seller_id}")
        # Plot each buyer's price as a dotted line
        for buyer in buyers:
            plt.plot(np.arange(len(buyer_price_history[buyer])), buyer_price_history[buyer], linestyle='dotted', color=color)

    plt.title("Price Adjustments Over Time (Sellers and Buyers)")
    plt.xlabel("Iterations")
    plt.ylabel("Price")
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    
    
def main_compare_convergence_connectivity():
    config = {
        "network_type": "random",
        "num_buyers": 30,
        "num_sellers": 5,
        "iterations": 100,
        "gamma": 1.0  # Elasticity parameter for buyer valuation
    }

    seller_config = {
        "seller_price_high": 100,
        "seller_price_low": 50,
        "seller_quantity_high": 20,
        "seller_quantity_low": 10
    }
    buyer_config = {
        "buyer_price_high": 50,
        "buyer_price_low": 20,
        "buyer_quantity_high": 20,
        "buyer_quantity_low": 10
    }

    connectivity_levels = [0.1, 0.2, 0.3, 0.5]  # Define different levels of connectivity
    results = []

    for connectivity_prob in connectivity_levels:
        print(f"Running simulation for connectivity: {connectivity_prob}")
        sellers = [Node(f"Seller_{i}", price=np.random.uniform(seller_config["seller_price_low"], seller_config["seller_price_high"]),
                        quantity=-np.random.uniform(seller_config["seller_quantity_low"], seller_config["seller_quantity_high"])) for i in range(config["num_sellers"])]

        buyers = [Node(f"Buyer_{i}", price=np.random.uniform(buyer_config["buyer_price_low"], buyer_config["buyer_price_high"]),
                       quantity=np.random.uniform(buyer_config["buyer_quantity_low"], buyer_config["buyer_quantity_high"]), gamma=config["gamma"]) for i in range(config["num_buyers"])]

        # Create networks with varying connectivity
        G_seller_buyer = create_random_market_network(buyers, sellers, connectivity_prob)
        check_and_fix_isolated_nodes(G_seller_buyer, buyers, sellers)
        G_buyer_buyer = create_buyer_buyer_network(buyers, sellers, G_seller_buyer)
        G_seller_seller = create_seller_seller_network(buyers, sellers, G_seller_buyer)

        # Initialize price history dictionaries
        price_history = {seller.id: [] for seller in sellers}
        buyer_price_history = {buyer.id: [] for buyer in buyers}

        iterations_without_opt_out = update_bids_with_utility_opt_out(G_seller_buyer, G_buyer_buyer, G_seller_seller, buyer_config, seller_config, price_history, buyer_price_history, use_opt_out=False)

        # Create seller-buyer connection dictionary
        #seller_buyer_connections = {seller.id: list(G_seller_buyer.neighbors(seller.id)) for seller in sellers}

        # Plot network and price history after simulation with opt-out
        #plot_network_and_price_history(price_history, buyer_price_history, G_seller_buyer, G_buyer_buyer, seller_buyer_connections)

        # Reset network and run without opt-out
        G_seller_buyer = create_random_market_network(buyers, sellers, connectivity_prob)
        check_and_fix_isolated_nodes(G_seller_buyer, buyers, sellers)
        G_buyer_buyer = create_buyer_buyer_network(buyers, sellers, G_seller_buyer)
        G_seller_seller = create_seller_seller_network(buyers, sellers, G_seller_buyer)
        
        price_history = {seller.id: [] for seller in sellers}  # Reset history
        buyer_price_history = {buyer.id: [] for buyer in buyers}  # Reset history
        
        # Run with opt-out
        iterations_with_opt_out = update_bids_with_utility_opt_out(G_seller_buyer, G_buyer_buyer, G_seller_seller, buyer_config, seller_config, price_history, buyer_price_history, use_opt_out=True)

        # Create seller-buyer connection dictionary
        #seller_buyer_connections = {seller.id: list(G_seller_buyer.neighbors(seller.id)) for seller in sellers}

        # Plot network and price history after simulation with opt-out
        #plot_network_and_price_history(price_history, buyer_price_history, G_seller_buyer, G_buyer_buyer, seller_buyer_connections)

        # Store the results for this connectivity level
        improvement_factor = iterations_without_opt_out / iterations_with_opt_out
        results.append({
            "connectivity": connectivity_prob,
            "iterations_with_opt_out": iterations_with_opt_out,
            "iterations_without_opt_out": iterations_without_opt_out,
            "improvement_factor": improvement_factor
        })

    # Print the results
    for result in results:
        print(f"Connectivity: {result['connectivity']}, "
              f"With opt-out: {result['iterations_with_opt_out']} iterations, "
              f"Without opt-out: {result['iterations_without_opt_out']} iterations, "
              f"Improvement factor: {result['improvement_factor']}")
              

main_compare_convergence_connectivity()
