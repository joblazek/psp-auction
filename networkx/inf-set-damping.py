import numpy as np
import networkx as nx
from collections import deque
import random
import matplotlib.pyplot as plt

np.random.seed()

# Node class representing a buyer or seller
class Node:
    def __init__(self, id, price, quantity):
        self.id = id
        self.price = price
        self.quantity = quantity  # Positive for buyers, negative for sellers
        self.bid = np.random.uniform(50, 100)  # Initial bid for buyers
        
def influence_function(node_id, G_seller_buyer, G_buyer_buyer, G_seller_seller, influence_threshold, window_size=5):
    """
    Calculates the damping factor for a node based on its influence in the network.

    :param node_id: ID of the node (buyer or seller)
    :param G_seller_buyer: Buyer-Seller network
    :param G_buyer_buyer: Buyer-Buyer network
    :param G_seller_seller: Seller-Seller network
    :param influence_threshold: Threshold for normalizing influence
    :param window_size: Time window size for calculating moving average
    :return: Damping factor (0 = no influence, 1 = maximum influence)
    """
    if node_id not in G_seller_buyer.nodes:
        raise ValueError(f"Node {node_id} not found in G_seller_buyer.")

    node = G_seller_buyer.nodes[node_id].get('obj', None)
    if node is None:
        raise ValueError(f"No 'obj' attribute found for node {node_id}.")

    # Initialize influence values
    influence_values = deque(maxlen=window_size)
    print(f"Processing node: {node_id}\n")
    print(f"Neighbors in G_seller_buyer: {list(G_seller_buyer.neighbors(node_id))}\n")
    print(f"Neighbors in G_seller_seller: {list(G_seller_seller.neighbors(node_id)) if 'Seller' in node_id else 'N/A'}\n")

    # Calculate influence based on node type
    if "Buyer" in node_id:
        # Buyers are influenced by seller prices
        seller_influence = [
            G_seller_buyer.nodes[neighbor]['obj'].price
            for neighbor in G_seller_buyer.neighbors(node_id)
            if "Seller" in neighbor and 'obj' in G_seller_buyer.nodes[neighbor]
        ]
        if seller_influence:
            avg_seller_influence = np.mean(seller_influence)
            influence_values.append(avg_seller_influence)

    elif "Seller" in node_id:
        # Sellers are influenced by buyer bids and neighboring sellers
        buyer_influence = [
            G_seller_buyer.nodes[neighbor]['obj'].bid
            for neighbor in G_seller_buyer.neighbors(node_id)
            if "Buyer" in neighbor and 'obj' in G_seller_buyer.nodes[neighbor]
        ]
        seller_influence = [
            G_seller_seller.nodes[neighbor]['obj'].price
            for neighbor in G_seller_seller.neighbors(node_id)
            if "Seller" in neighbor and 'obj' in G_seller_seller.nodes[neighbor]
        ]

        if buyer_influence:
            avg_buyer_influence = np.mean(buyer_influence)
            influence_values.append(avg_buyer_influence)
        if seller_influence:
            avg_seller_influence = np.mean(seller_influence)
            influence_values.append(avg_seller_influence)

    # Compute the damping factor
    if influence_values:
        avg_influence = np.mean(influence_values)
        damping_factor = min(avg_influence / influence_threshold, 1)  # Normalize and cap at 1
    else:
        damping_factor = 0  # No influence detected
    
    print(f"Influence values: {influence_values}\n")
    print(f"\n\n\n")

    return damping_factor
    
# Calculate utility with damping
def calculate_utility_with_damping(buyer, seller, damping_factor):
    valuation = buyer.quantity * (1 / (seller.price + 1e-9))
    return valuation * damping_factor

# Update influence sets by adjusting edge weights
def calculate_influence_sets_with_damping(G_seller_buyer, damping_factors):
    """
    Adjust edge weights in G_seller_buyer based on damping factors.
    """
    for seller_id in [n for n in G_seller_buyer if "Seller" in n]:
        for buyer_id in G_seller_buyer.neighbors(seller_id):
            weight = G_seller_buyer[seller_id][buyer_id].get('weight', 1.0)
            adjusted_weight = weight * damping_factors.get(seller_id, 1.0)
            G_seller_buyer[seller_id][buyer_id]['weight'] = adjusted_weight

# Update bids and prices with damping factors
def update_bids_psp_proportional_with_influence(G_seller_buyer, G_buyer_buyer, G_seller_seller, buyer_config, seller_config, influence_threshold):
    """
    Update bids and prices based on the PSP mechanism and damping factors.
    """
    damping_factors = {}

    # Calculate damping factors for all nodes in the graph
    for node_id in G_seller_buyer.nodes:
        damping_factors[node_id] = influence_function(node_id, G_seller_buyer, G_buyer_buyer, G_seller_seller, influence_threshold)

    # Ensure all nodes have damping factors
    for node_id in G_seller_buyer.nodes:
        if node_id not in damping_factors:
            damping_factors[node_id] = 1.0  # Default damping factor

    calculate_influence_sets_with_damping(G_seller_buyer, damping_factors)

    for node_id in G_seller_buyer.nodes:
        node = G_seller_buyer.nodes[node_id]['obj']
        damping_factor = damping_factors[node_id]

        if "Seller" in node_id:
            buyer_bids = [G_seller_buyer.nodes[neighbor]['obj'].bid for neighbor in G_seller_buyer.neighbors(node_id) if "Buyer" in neighbor]
            if len(buyer_bids) > 1:
                second_highest_bid = sorted(buyer_bids)[-2]
                node.price = node.price + damping_factor * (second_highest_bid - node.price)

            connected_seller_prices = [G_seller_seller.nodes[neighbor]['obj'].price for neighbor in G_seller_seller.neighbors(node_id)]
            if connected_seller_prices:
                avg_seller_price = np.mean(connected_seller_prices)
                node.price = node.price + damping_factor * (avg_seller_price - node.price)

        elif "Buyer" in node_id:
            seller_prices = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_buyer.neighbors(node_id) if "Seller" in neighbor]
            if seller_prices:
                min_seller_price = min(seller_prices)
                node.bid = node.bid + damping_factor * (min_seller_price - node.bid)
                

# Example network setup
def setup_network(num_buyers, num_sellers):
    buyers = [Node(f"Buyer_{i}", price=0, quantity=np.random.uniform(10, 20)) for i in range(num_buyers)]
    sellers = [Node(f"Seller_{i}", price=np.random.uniform(50, 100), quantity=-np.random.uniform(10, 20)) for i in range(num_sellers)]
    G_seller_buyer = nx.Graph()

    for seller in sellers:
        G_seller_buyer.add_node(seller.id, obj=seller)
    for buyer in buyers:
        G_seller_buyer.add_node(buyer.id, obj=buyer)

    for seller in sellers:
        connected_buyers = random.sample(buyers, k=random.randint(1, len(buyers)))
        for buyer in connected_buyers:
            G_seller_buyer.add_edge(seller.id, buyer.id)

    return G_seller_buyer, buyers, sellers
    
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
    
# Store price history and note when participants exit the market
def track_price_history(price_history, buyer_price_history, buyers, sellers, iteration):
    for seller in sellers:
        if seller.id not in price_history:
            price_history[seller.id] = [None] * iteration  # Fill in past with None or NaN
        price_history[seller.id].append(seller.price)
    
    for buyer in buyers:
        if buyer.id not in buyer_price_history:
            buyer_price_history[buyer.id] = [None] * iteration  # Fill in past with None or NaN
        buyer_price_history[buyer.id].append(buyer.bid)


    
# Main simulation function
def run_simulation():
    num_buyers, num_sellers = 10, 5
    G_seller_buyer, buyers, sellers = setup_network(num_buyers, num_sellers)
    
    G_buyer_buyer = create_buyer_buyer_network(buyers, sellers, G_seller_buyer)
    G_seller_seller = create_seller_seller_network(buyers, sellers, G_seller_buyer)
    
    influence_threshold = 50.0
    iterations = 20
    
    price_history = {seller.id: [] for seller in sellers}
    buyer_price_history = {buyer.id: [] for buyer in buyers}


    # Run the simulation for the specified number of iterations
    for iteration in range(iterations):
        print(f"Iteration {iteration}.\n\n")

        update_bids_psp_proportional_with_influence(G_seller_buyer, G_buyer_buyer, G_seller_seller, {}, {}, influence_threshold)
    
        # Track price history for sellers and buyers
        track_price_history(price_history, buyer_price_history, buyers, sellers, iteration)
    
    # Visualize the final results
    seller_buyer_connections = {seller.id: list(G_seller_buyer.neighbors(seller.id)) for seller in sellers}
    plot_network_and_price_history(price_history, buyer_price_history, G_seller_buyer, G_buyer_buyer, seller_buyer_connections)

run_simulation()
