import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import json

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

    if isolated_buyers:
        print(f"Warning: Isolated buyers detected: {isolated_buyers}. Adding random seller connections.")
        for buyer_id in isolated_buyers:
            # Randomly connect the isolated buyer to one or more sellers
            random_seller = random.choice(sellers)
            G_seller_buyer.add_edge(random_seller.id, buyer_id)

    return G_seller_buyer
    
    
# Update the price logic to include influence from the seller-seller network
def update_bids_psp(G_seller_buyer, G_buyer_buyer, G_seller_seller):
    for seller_id in G_seller_buyer.nodes:
        node = G_seller_buyer.nodes[seller_id]['obj']
        
        # Seller logic: Adjust based on connected buyer bids and seller-seller influence
        if "Seller" in seller_id:  # Seller
            buyer_bids = [G_seller_buyer.nodes[neighbor]['obj'].bid for neighbor in G_seller_buyer.neighbors(seller_id)]
            if len(buyer_bids) > 1:
                second_highest_bid = sorted(buyer_bids)[-2]  # Progressive Second Price: Second-highest bid
                node.price = second_highest_bid  # Sellers price at second-highest buyer bid

            # Seller-Seller influence: Adjust based on prices of connected sellers
            connected_seller_prices = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_seller.neighbors(seller_id)]
            if connected_seller_prices:
                avg_seller_price = np.mean(connected_seller_prices)
                # Mix current price with the average price of connected sellers
                node.price = (node.price + avg_seller_price) / 2

        # Buyer logic: Adjust based on connected sellers and other buyers (PSP influenced)
        elif "Buyer" in seller_id:  # Buyer
            seller_prices = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_buyer.neighbors(seller_id) if "Seller" in neighbor]
            if seller_prices:
                min_seller_price = min(seller_prices)  # Buyers prefer the lowest price
                # Adjust bid based on both seller prices and competition from other buyers
                other_buyer_bids = [G_buyer_buyer.nodes[neighbor]['obj'].bid for neighbor in G_buyer_buyer.neighbors(seller_id)]
                if other_buyer_bids:
                    second_highest_buyer_bid = sorted(other_buyer_bids)[-2] if len(other_buyer_bids) > 1 else other_buyer_bids[0]
                    G_seller_buyer.nodes[seller_id]['obj'].bid = (min_seller_price + second_highest_buyer_bid) / 2  # Adjust bid competitively

# Calculate Supply-Demand Ratio (SDR)
def calculate_sdr(buyers, sellers):
    total_supply = abs(sum(seller.quantity for seller in sellers))
    total_demand = sum(buyer.quantity for buyer in buyers)
    if total_demand == 0:
        return float('inf')
    return total_supply / total_demand

# OU Process for adjusting SDR
def ou_process_sdr(SDR, mu, theta, sigma, dt):
    dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion increment
    SDR += theta * (mu - SDR) * dt + sigma * dW
    return SDR

# Adjust market participants by reinitializing buyers and sellers
def adjust_market_participants(SDR, buyers, sellers, SDR_threshold_high, SDR_threshold_low, buyer_config, seller_config):
    if SDR > SDR_threshold_high:  # Too much supply            
        if len(buyers) > 0:    
            buyer_price_high = buyer_config["buyer_price_high"]
            buyer_price_low = buyer_config["buyer_price_low"]
            buyer_quantity_high = buyer_config["buyer_quantity_high"]
            buyer_quantity_low = buyer_config["buyer_quantity_low"] 
            # Reinitialize a buyer
            buyer_to_reset = random.choice(buyers)
            buyer_to_reset.price = np.random.uniform(buyer_price_low, buyer_price_high)
            buyer_to_reset.quantity = np.random.uniform(buyer_quantity_low, buyer_quantity_high)
            print(f"Reinitialized Buyer: {buyer_to_reset.id} with new bid {buyer_to_reset.price} and quantity {buyer_to_reset.quantity}")

    elif SDR < SDR_threshold_low:  # Too much demand      
        if len(sellers) > 0:
            
            seller_price_high = seller_config["seller_price_high"]
            seller_price_low = seller_config["seller_price_low"]
            seller_quantity_high = seller_config["seller_quantity_high"]
            seller_quantity_low = seller_config["seller_quantity_low"]
            # Reinitialize a seller
            seller_to_reset = random.choice(sellers)
            seller_to_reset.price = np.random.uniform(seller_price_low, seller_price_high)
            seller_to_reset.quantity = -np.random.uniform(seller_quantity_low, seller_quantity_high)
            print(f"Reinitialized Seller: {seller_to_reset.id} with new price {seller_to_reset.price} and quantity {seller_to_reset.quantity}")
            
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
    plt.show()
    
    
# Modify market participants dynamically based on their fulfilled demand or supply
def proportional_allocation(buyers, sellers, buyer_config, seller_config):
    
    # Buyers
    buyer_price_high = buyer_config["buyer_price_high"]
    buyer_price_low = buyer_config["buyer_price_low"]
    buyer_quantity_high = buyer_config["buyer_quantity_high"]
    buyer_quantity_low = buyer_config["buyer_quantity_low"] 
    # Remove buyers that have satisfied their demand
    satisfied_buyers = [buyer for buyer in buyers if buyer.quantity <= 0]
    for buyer in satisfied_buyers:
        buyer.price = np.random.uniform(buyer_price_low, buyer_price_high)
        buyer.quantity = np.random.uniform(buyer_quantity_low, buyer_quantity_high)
        print(f"Buyer {buyer.id} has satisfied their demand and left the market.")
        print(f"Reinitialized Buyer: {buyer.id} with new bid {buyer.price} and quantity {buyer.quantity}")
    
    # Sellers
    seller_price_high = seller_config["seller_price_high"]
    seller_price_low = seller_config["seller_price_low"]
    seller_quantity_high = seller_config["seller_quantity_high"]
    seller_quantity_low = seller_config["seller_quantity_low"]
    # Remove sellers that have sold all their goods
    satisfied_sellers = [seller for seller in sellers if seller.quantity >= 0]
    for seller in satisfied_sellers:
        seller.price = np.random.uniform(seller_price_low, seller_price_high)
        seller.quantity = -np.random.uniform(seller_quantity_low, seller_quantity_high)
        print(f"Reinitialized Seller: {seller.id} with new price {seller.price} and quantity {seller.quantity}")
        print(f"Seller {seller.id} has sold all their goods and left the market.")
        
        
# Modify bids and prices based on real-time competition and proportional allocation
def update_bids_psp_proportional(G_seller_buyer, G_buyer_buyer, G_seller_seller, buyer_config, seller_config):
    for seller_id in G_seller_buyer.nodes:
        node = G_seller_buyer.nodes[seller_id]['obj']
        
        # Seller logic: Adjust based on connected buyer bids and seller-seller influence
        if "Seller" in seller_id:  # Seller
            buyer_bids = [G_seller_buyer.nodes[neighbor]['obj'].bid for neighbor in G_seller_buyer.neighbors(seller_id)]
            if len(buyer_bids) > 1:
                second_highest_bid = sorted(buyer_bids)[-2]  # PSP: Second-highest bid
                node.price = second_highest_bid  # Sellers price at second-highest buyer bid

            # Seller-Seller influence: Adjust based on prices of connected sellers
            connected_seller_prices = [G_seller_seller.nodes[neighbor]['obj'].price for neighbor in G_seller_seller.neighbors(seller_id)]
            if connected_seller_prices:
                avg_seller_price = np.mean(connected_seller_prices)
                # Sellers adjust to be competitive (mix current price with average of connected sellers)
                node.price = (node.price + avg_seller_price) / 2
                
            # Increase seller's quantity by an amount proportional to the bid
            node.quantity += node.bid / second_highest_bid  # Buyer buys proportional to the bid
            
        # Buyer logic: Adjust bids based on connected sellers and other buyers
        elif "Buyer" in seller_id:  # Buyer
            seller_prices = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_buyer.neighbors(seller_id) if "Seller" in neighbor]
            if seller_prices:
                min_seller_price = min(seller_prices)  # Buyers prefer the lowest price
                other_buyer_bids = [G_buyer_buyer.nodes[neighbor]['obj'].bid for neighbor in G_buyer_buyer.neighbors(seller_id)]
                if other_buyer_bids:
                    second_highest_buyer_bid = sorted(other_buyer_bids)[-2] if len(other_buyer_bids) > 1 else other_buyer_bids[0]
                    # Buyers bid competitively based on other buyers and sellers' prices
                    G_seller_buyer.nodes[seller_id]['obj'].bid = (min_seller_price + second_highest_buyer_bid) / 2

            # Reduce buyer's quantity by an amount proportional to the bid
            node.quantity -= node.bid / min_seller_price  # Buyer buys proportional to the bid
                            
        # After the bid updates, check if any participants can leave the market
        proportional_allocation([G_seller_buyer.nodes[buyer]['obj'] for buyer in G_seller_buyer if 'Buyer' in buyer],
                            [G_seller_buyer.nodes[seller]['obj'] for seller in G_seller_buyer if 'Seller' in seller], buyer_config, seller_config)
                            
                            
# Local influence of SDR
def update_seller_prices_local_only(G_seller_buyer, G_seller_seller):
    for seller_id in G_seller_buyer.nodes:
        node = G_seller_buyer.nodes[seller_id]['obj']
        
        if "Seller" in seller_id:
            # Step 1: Influence from Buyer Bids (Second-Highest Bid)
            buyer_bids = [G_seller_buyer.nodes[neighbor]['obj'].bid for neighbor in G_seller_buyer.neighbors(seller_id)]
            if len(buyer_bids) > 1:
                second_highest_bid = sorted(buyer_bids)[-2]
                local_price = second_highest_bid

            # Step 2: Influence from Neighboring Sellers
            connected_seller_prices = [G_seller_seller.nodes[neighbor]['obj'].price for neighbor in G_seller_seller.neighbors(seller_id)]
            if connected_seller_prices:
                avg_seller_price = np.mean(connected_seller_prices)
                # Combine local price with neighboring sellers' prices to remain competitive
                local_price = (local_price + avg_seller_price) / 2

            # Step 3: Inventory-Driven Adjustment
            if node.quantity < 0:
                # If the seller has unsold goods (negative quantity), reduce the price
                inventory_factor = max(0.9, (1 + node.quantity / 20))  # The more unsold goods, the bigger the reduction
                node.price = local_price * inventory_factor
            else:
                # If the seller is nearly sold out, they can raise their price slightly
                node.price = local_price * 1.05  # Small increase for sellers with little remaining inventory
                
                
# Main function to run the simulation with proportional allocation
def main_network_simulation_proportional():
    config = {
    "network_type": "",
    "num_buyers": 5,
    "num_sellers": 2,
    "mu": 1.0, # Long-term mean (equilibrium SDR)
    "theta": 0.1, # Speed of reversion to the mean
    "sigma": 1.0, # Volatility (how much SDR can change at each step)
    "iterations": 100,
    "num_clusters": 2,
    "SDR_threshold_high": 1.7,
    "SDR_threshold_low": 0.3,
    "dt": 1, # Time increment
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
    
    # Initialize SDR array
    SDR = np.zeros(iterations+1)
    
    # Range SDR
    mu = np.linspace(0,2,iterations+1)

    network_type = config["network_type"]
    num_buyers = config["num_buyers"]
    num_sellers = config["num_sellers"]
    #mu = config["mu"]
    theta = config["theta"]
    sigma = config["sigma"]
    SDR_threshold_high = config["SDR_threshold_high"]
    SDR_threshold_low = config["SDR_threshold_low"]
    
    # Sellers
    seller_price_high = seller_config["seller_price_high"]
    seller_price_low = seller_config["seller_price_low"]
    seller_quantity_high = seller_config["seller_quantity_high"]
    seller_quantity_low = seller_config["seller_quantity_low"]
    sellers = [Node(f"Seller_{i}", price=np.random.uniform(seller_price_low, seller_price_high), quantity=-np.random.uniform(seller_quantity_low, seller_quantity_high)) for i in range(num_sellers)]
    
    # Buyers
    buyer_price_high = buyer_config["buyer_price_high"]
    buyer_price_low = buyer_config["buyer_price_low"]
    buyer_quantity_high = buyer_config["buyer_quantity_high"]
    buyer_quantity_low = buyer_config["buyer_quantity_low"] 
    buyers = [Node(f"Buyer_{i}", price=np.random.uniform(buyer_price_low, buyer_price_high), quantity=np.random.uniform(buyer_quantity_low, buyer_quantity_high)) for i in range(num_buyers)]
 
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

        # Ensure price history for any new sellers and buyers
        for seller in sellers:
            if seller.id not in price_history:
                price_history[seller.id] = []
        for buyer in buyers:
            if buyer.id not in buyer_price_history:
                buyer_price_history[buyer.id] = []

        # Apply Progressive Second Price (PSP) logic <--- nested function is proportional
        update_bids_psp_proportional(G_seller_buyer, G_buyer_buyer, G_seller_seller, buyer_config, seller_config)
                            
        # Calculate SDR
#        SDR[iteration] = calculate_sdr(buyers, sellers)

        # Adjust SDR using OU process
#        SDR[iteration] = ou_process_sdr(SDR[iteration], mu[iteration], theta, sigma, dt)

        # Local pricing rule
        #update_seller_prices_local_only(G_seller_buyer, G_seller_seller)

        # Adjust market participants based on SDR
#        adjust_market_participants(SDR[iteration], buyers, sellers, SDR_threshold_high, SDR_threshold_low)
        
        # Track price history for sellers and buyers
        track_price_history(price_history, buyer_price_history, buyers, sellers, iteration)

    # Visualize the final results
    seller_buyer_connections = {seller.id: list(G_seller_buyer.neighbors(seller.id)) for seller in sellers}
    plot_network_and_price_history(price_history, buyer_price_history, G_seller_buyer, G_buyer_buyer, seller_buyer_connections)


main_network_simulation_proportional()
