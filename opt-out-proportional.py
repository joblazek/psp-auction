import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import json

from collections import deque

np.random.seed()


# Node class representing a buyer or seller
class Node:
    def __init__(self, id, price, quantity):
        self.id = id
        self.price = price
        self.quantity = quantity  # Positive for buyers, negative for sellers
        self.bid = np.random.uniform(50, 100)  # Initial bid for buyers

# Create a network where buyers are connected to multiple sellers
def create_random_market_network(buyers, sellers):
    print("Creating random network")
    G_seller_buyer = nx.Graph()

    for seller in sellers:
        G_seller_buyer.add_node(seller.id, obj=seller)
    for buyer in buyers:
        G_seller_buyer.add_node(buyer.id, obj=buyer)

    # Randomly connect buyers to sellers
    for seller in sellers:
        connected_buyers = random.sample(buyers, k=random.randint(1, len(buyers)))
        for buyer in connected_buyers:
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
    
    
# Create a monopoly buyer network
def create_monopoly_buyer_network(buyers, sellers):
    print("Creating monopoly buyer network")
    G_seller_buyer = nx.Graph()
    
    # One buyer (the "monopoly" buyer) connected to all sellers
    monopoly_buyer = buyers[0]
    G_seller_buyer.add_node(monopoly_buyer.id, obj=monopoly_buyer)
    for seller in sellers:
        G_seller_buyer.add_node(seller.id, obj=seller)
        G_seller_buyer.add_edge(seller.id, monopoly_buyer.id)

    # Other buyers are only connected to one seller each
    for i, buyer in enumerate(buyers[1:]):
        G_seller_buyer.add_node(buyer.id, obj=buyer)
        G_seller_buyer.add_edge(sellers[i % len(sellers)].id, buyer.id)

    return G_seller_buyer

# Create isolated buyers network
def create_isolated_buyers_network(buyers, sellers):
    print("Creating isolated buyer network")
    G_seller_buyer = nx.Graph()

    for i, buyer in enumerate(buyers):
        seller = sellers[i % len(sellers)]
        G_seller_buyer.add_node(buyer.id, obj=buyer)
        G_seller_buyer.add_node(seller.id, obj=seller)
        G_seller_buyer.add_edge(seller.id, buyer.id)

    return G_seller_buyer

# Create monopoly seller network
def create_monopoly_seller_network(buyers, sellers):
    print("Creating monopoly seller network")
    G_seller_buyer = nx.Graph()

    monopoly_seller = sellers[0]
    G_seller_buyer.add_node(monopoly_seller.id, obj=monopoly_seller)
    for buyer in buyers:
        G_seller_buyer.add_node(buyer.id, obj=buyer)
        G_seller_buyer.add_edge(monopoly_seller.id, buyer.id)

    # Other sellers are only connected to one buyer each
    for i, seller in enumerate(sellers[1:]):
        G_seller_buyer.add_node(seller.id, obj=seller)
        G_seller_buyer.add_edge(seller.id, buyers[i % len(buyers)].id)

    return G_seller_buyer

# Create clustered subgroups network
def create_clustered_subgroups_network(buyers, sellers, num_clusters=2):
    print("Creating clustered subgroups network")
    G_seller_buyer = nx.Graph()
    
    # Split buyers and sellers into clusters
    clusters = []
    for i in range(num_clusters):
        buyer_cluster = buyers[i::num_clusters]
        seller_cluster = sellers[i::num_clusters]
        clusters.append((buyer_cluster, seller_cluster))

    # Create edges within each cluster
    for buyer_cluster, seller_cluster in clusters:
        for buyer in buyer_cluster:
            for seller in seller_cluster:
                G_seller_buyer.add_node(buyer.id, obj=buyer)
                G_seller_buyer.add_node(seller.id, obj=seller)
                G_seller_buyer.add_edge(seller.id, buyer.id)

    return G_seller_buyer

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
    

# Update neighbors after node reinitialization
def update_neighbors_after_reinitialization(node_id, node_type, buyers, sellers, G_seller_buyer):
    # Remove the current edges for the reinitialized node
    G_seller_buyer.remove_edges_from(list(G_seller_buyer.edges(node_id)))

    # Add new edges based on the node type
    if node_type == "buyer":
        # Reassign a new set of sellers for the buyer
        new_sellers = random.sample(sellers, k=random.randint(1, len(sellers)))
        for seller in new_sellers:
            G_seller_buyer.add_edge(node_id, seller.id)

    elif node_type == "seller":
        # Reassign a new set of buyers for the seller
        new_buyers = random.sample(buyers, k=random.randint(2, len(buyers)))
        for buyer in new_buyers:
            G_seller_buyer.add_edge(node_id, buyer.id)

    return G_seller_buyer

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
    #plt.show()
    
    
    # Save the figure as an SVG file
    output_path = "/storage/emulated/0/Download/clustered_network.pdf"  # Change this path
    plt.savefig(output_path, format="pdf", dpi=300)  # Save as SVG
    plt.show()
    
    print(f"Figure saved to {output_path}")
    
    
# Modify market participants dynamically based on their fulfilled demand or supply
def proportional_allocation(G_seller_buyer, buyers, sellers, buyer_config, seller_config):
        
        reinitialized = False  # Flag to track if any reinitialization happens
        sph = seller_config["seller_price_high"]
        spl = seller_config["seller_price_low"]
        sqh = seller_config["seller_quantity_high"]
        sql = seller_config["seller_quantity_low"]
        bph = buyer_config["buyer_price_high"]
        bpl = buyer_config["buyer_price_low"]
        bqh = buyer_config["buyer_quantity_high"]
        bql = buyer_config["buyer_quantity_low"] 
        
        # Remove buyers that have satisfied their demand
        satisfied_buyers = [buyer for buyer in buyers if buyer.quantity <= 0]
        for buyer in satisfied_buyers:
            buyer.price = np.random.uniform(bpl, bph)
            buyer.quantity = np.random.uniform(bql, bqh)
            print(f"Buyer {buyer.id} has satisfied their demand and left the market.")
            G_seller_buyer = update_neighbors_after_reinitialization(buyer.id, "buyer", buyers, sellers, G_seller_buyer)
            reinitialized = True
    
        # Remove sellers that have sold all their goods
        satisfied_sellers = [seller for seller in sellers if seller.quantity >= 0]
        for seller in satisfied_sellers:
            seller.price = np.random.uniform(spl, sph)
            seller.quantity = -np.random.uniform(sql, sqh)
            print(f"Seller {seller.id} has sold all their goods and left the market.")
            G_seller_buyer = update_neighbors_after_reinitialization(seller.id, "seller", buyers, sellers, G_seller_buyer)
            reinitialized = True
            
        return reinitialized
                            
# Utility calculation for buyers
def calculate_utility(buyer, allocation, price):
    valuation = buyer.valuation_function(allocation)
    cost = allocation * price
    return valuation - cost

# Opt-out function based on utility maximization
def opt_out_utility_based(buyer, connected_sellers, G_seller_buyer, gamma=1.0):
    sellers_sorted = sorted(connected_sellers, key=lambda s: s.price)
    total_demand = buyer.quantity
    total_allocated = 0
    chosen_sellers = []

    for seller in sellers_sorted:
        allocation = min(abs(seller.quantity), total_demand - total_allocated)
        price = seller.price
        utility = calculate_utility(buyer, allocation, price)
        if utility > 0:
            chosen_sellers.append(seller)
            total_allocated += allocation
        if total_allocated >= total_demand:
            break

    return chosen_sellers
    

# Opt-out function that checks if a node should opt out based on influence sets
def opt_out_function(node_id, G_seller_buyer, G_buyer_buyer, G_seller_seller, influence_threshold, window_size=5):
    """
    Determines if a node (buyer or seller) should opt out of the market.
    
    :param node_id: ID of the node (buyer or seller)
    :param G_seller_buyer: Buyer-Seller network
    :param G_buyer_buyer: Buyer-Buyer network
    :param G_seller_seller: Seller-Seller network
    :param influence_threshold: Threshold below which the node opts out
    :param window_size: Time window size for calculating moving average influence
    :return: True if node opts out, False otherwise
    """
    # Get the node object
    node = G_seller_buyer.nodes[node_id]['obj']
    
    # Initialize influence sets
    influence_set = deque(maxlen=window_size)
    
    # Calculate influence from buyer-seller connections
    if "Buyer" in node_id:
        # Buyers are influenced by sellers
        seller_influence = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_buyer.neighbors(node_id) if "Seller" in neighbor]
        if seller_influence:
            avg_seller_influence = np.mean(seller_influence)
            influence_set.append(avg_seller_influence)
    
    elif "Seller" in node_id:
        # Sellers are influenced by buyers and other sellers
        buyer_influence = [G_seller_buyer.nodes[neighbor]['obj'].bid for neighbor in G_seller_buyer.neighbors(node_id) if "Buyer" in neighbor]
        seller_influence = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_seller.neighbors(node_id)]
        
        if buyer_influence:
            avg_buyer_influence = np.mean(buyer_influence)
            influence_set.append(avg_buyer_influence)
        if seller_influence:
            avg_seller_influence = np.mean(seller_influence)
            influence_set.append(avg_seller_influence)
    
    # Calculate the moving average of the influence set
    if len(influence_set) == window_size:
        avg_influence = np.mean(influence_set)
        # Opt-out if the influence is below the threshold
        if avg_influence < influence_threshold:
            print(f"Node {node_id} opts out due to low influence: {avg_influence}")
            return True
    
    return False


def update_bids_psp_proportional_with_opt_out(G_seller_buyer, G_buyer_buyer, G_seller_seller, buyer_config, seller_config, influence_threshold):
    """
    Combined PSP logic with proportional allocation and seller-seller influence.
    Updates bids and prices for buyers and sellers, and considers opt-out based on influence sets.
    
    :param G_seller_buyer: Buyer-Seller network
    :param G_buyer_buyer: Buyer-Buyer network
    :param G_seller_seller: Seller-Seller network
    :param buyer_config: Configuration for buyers
    :param seller_config: Configuration for sellers
    :param influence_threshold: Threshold for opt-out decision
    """
    
    for node_id in G_seller_buyer.nodes:
        node = G_seller_buyer.nodes[node_id]['obj']
        
        # Opt-out function: Skip this node if the influence threshold is met
        if opt_out_function(node_id, G_seller_buyer, G_buyer_buyer, G_seller_seller, influence_threshold):
            continue  # Skip this node if it opts out

        # Logic for Sellers: Adjust prices based on bids and seller-seller influence
        if "Seller" in node_id:
            buyer_bids = [G_seller_buyer.nodes[neighbor]['obj'].bid for neighbor in G_seller_buyer.neighbors(node_id) if "Buyer" in neighbor]
            if len(buyer_bids) > 1:
                second_highest_bid = sorted(buyer_bids)[-2]  # PSP: Second-highest bid
                node.price = second_highest_bid  # Sellers price at second-highest buyer bid

            # Seller-Seller influence: Adjust based on prices of connected sellers
            connected_seller_prices = [G_seller_seller.nodes[neighbor]['obj'].price for neighbor in G_seller_seller.neighbors(node_id)]
            if connected_seller_prices:
                avg_seller_price = np.mean(connected_seller_prices)
                node.price = (node.price + avg_seller_price) / 2  # Adjust price based on seller neighbors

            # Proportional Allocation: Increase seller's quantity proportional to bids
            if len(buyer_bids) > 1:
                node.quantity += node.bid / second_highest_bid  # Buyer buys proportional to the bid

        # Logic for Buyers: Adjust bids based on seller prices and competition
        elif "Buyer" in node_id:
            seller_prices = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_buyer.neighbors(node_id) if "Seller" in neighbor]
            if seller_prices:
                min_seller_price = min(seller_prices)  # Buyers prefer the lowest seller price
                other_buyer_bids = [G_buyer_buyer.nodes[neighbor]['obj'].bid for neighbor in G_buyer_buyer.neighbors(node_id)]
                
                # If there are competing buyers, adjust bids competitively (PSP logic)
                if other_buyer_bids:
                    second_highest_buyer_bid = sorted(other_buyer_bids)[-2] if len(other_buyer_bids) > 1 else other_buyer_bids[0]
                    node.bid = (min_seller_price + second_highest_buyer_bid) / 2  # PSP influenced bid

            # Proportional Allocation: Reduce buyer's quantity based on their bid
            if min_seller_price > 0:
                node.quantity -= node.bid / min_seller_price  # Buyer buys proportional to the bid
                

def save_output_to_file(price_history, buyer_price_history, filename='output.json'):
    """
    Save price history and buyer price history to a JSON file.
    
    :param price_history: Seller price history
    :param buyer_price_history: Buyer price history
    :param filename: Output filename (default 'output.json')
    """
    output_data = {
        'price_history': price_history,
        'buyer_price_history': buyer_price_history
    }
    
    # Write the output to a file in JSON format
    with open(filename, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)
    
    print(f"Output saved to {filename}")


# Main function to run the simulation with proportional allocation, PSP logic and opt-out based on influence sets
def main_network_simulation_psp_proportional_with_opt_out():
    config = {
    "network_type": "clustered_subgroups",
    "num_buyers": 20,
    "num_sellers": 5,
     "iterations": 20,
    "num_clusters": 5,
    "influence_threshold": 50.0,  # Example threshold for opting out
    "dt": 1 # Time increment
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
    "buyer_quantity_low": 10,
    }
    
    # Get size for storage arrays
    iterations = config["iterations"]
    dt = config["dt"]
    

    network_type = config["network_type"]
    num_buyers = config["num_buyers"]
    num_sellers = config["num_sellers"]
    
    
    # Sellers
    sph = seller_config["seller_price_high"]
    spl = seller_config["seller_price_low"]
    sqh = seller_config["seller_quantity_high"]
    sql = seller_config["seller_quantity_low"]
    sellers = [Node(f"Seller_{i}", price=np.random.uniform(spl, sph), quantity=-np.random.uniform(sql, sqh)) for i in range(num_sellers)]
    
    # Buyers
    bph = buyer_config["buyer_price_high"]
    bpl = buyer_config["buyer_price_low"]
    bqh = buyer_config["buyer_quantity_high"]
    bql = buyer_config["buyer_quantity_low"] 
    buyers = [Node(f"Buyer_{i}", price=np.random.uniform(bpl, bph), quantity=np.random.uniform(bql, bqh)) for i in range(num_buyers)]
 
    price_history = {seller.id: [] for seller in sellers}
    buyer_price_history = {buyer.id: [] for buyer in buyers}

    # Network creation (depending on type)
    if network_type == 'monopoly_buyer':
        G_seller_buyer = create_monopoly_buyer_network(buyers, sellers)
    elif network_type == 'isolated_buyers':
        G_seller_buyer = create_isolated_buyers_network(buyers, sellers)
    elif network_type == 'monopoly_seller':
        G_seller_buyer = create_monopoly_seller_network(buyers, sellers)
    elif network_type == 'clustered_subgroups':
        G_seller_buyer = create_clustered_subgroups_network(buyers, sellers, config["num_clusters"])
    else:
        G_seller_buyer = create_random_market_network(buyers, sellers)
        check_and_fix_isolated_nodes(G_seller_buyer, buyers, sellers)

    # Create buyer-buyer and seller-seller networks
    G_buyer_buyer = create_buyer_buyer_network(buyers, sellers, G_seller_buyer)
    G_seller_seller = create_seller_seller_network(buyers, sellers, G_seller_buyer)

    # Run the simulation for the specified number of iterations
    for iteration in range(iterations):
        print(f"Iteration {iteration}.\n\n")

        # Update bids with PSP logic, proportional allocation and opt-out based on influence sets
        update_bids_psp_proportional_with_opt_out(G_seller_buyer, G_buyer_buyer, G_seller_seller, buyer_config, seller_config, config['influence_threshold'])
        
        # Track price history for sellers and buyers
        track_price_history(price_history, buyer_price_history, buyers, sellers, iteration)
        
        buyers = [G_seller_buyer.nodes[buyer]['obj'] for buyer in G_seller_buyer if 'Buyer' in buyer]
        sellers = [G_seller_buyer.nodes[seller]['obj'] for seller in G_seller_buyer if 'Seller' in seller]
        
        # After the bid updates, check if any participants can leave the market
        reinitialized = proportional_allocation(G_seller_buyer, buyers, sellers, buyer_config, seller_config)
        
        # Rebuild only the buyer-buyer and seller-seller networks based on the updated buyer-seller network
        if reinitialized:
            check_and_fix_isolated_nodes(G_seller_buyer, buyers, sellers)
            G_buyer_buyer = create_buyer_buyer_network(buyers, sellers, G_seller_buyer)
            G_seller_seller = create_seller_seller_network(buyers, sellers, G_seller_buyer)

    # Visualize the final results
    seller_buyer_connections = {seller.id: list(G_seller_buyer.neighbors(seller.id)) for seller in sellers}
    plot_network_and_price_history(price_history, buyer_price_history, G_seller_buyer, G_buyer_buyer, seller_buyer_connections)
    save_output_to_file(price_history, buyer_price_history, 'market_simulation_output.json')



main_network_simulation_psp_proportional_with_opt_out()
