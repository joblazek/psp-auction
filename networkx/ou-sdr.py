import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

np.random.seed()

# Node class representing a buyer or seller
class Node:
    def __init__(self, id, price):
        self.id = id
        self.price = price  # Initial price for sellers
        self.bid = np.random.uniform(50, 100)  # Initial bid for buyers

# Ornstein-Uhlenbeck process for SDR
def ou_process_sdr(SDR, mu, theta, sigma, dt):
    dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion increment
    SDR += theta * (mu - SDR) * dt + sigma * dW
    return max(0, SDR)  # Ensure SDR is non-negative

# Update prices and bids based on SDR
def update_prices_and_bids_with_sdr(G_seller_buyer, SDR):
    for seller_id in G_seller_buyer.nodes:
        node = G_seller_buyer.nodes[seller_id]['obj']

        if "Seller" in seller_id:
            # Seller's price scales with SDR
            buyer_bids = [G_seller_buyer.nodes[neighbor]['obj'].bid for neighbor in G_seller_buyer.neighbors(seller_id)]
            if len(buyer_bids) > 1:
                second_highest_bid = sorted(buyer_bids)[-2]
                node.price = second_highest_bid * SDR  # Sellers price proportional to SDR

        elif "Buyer" in seller_id:
            # Buyer's bid inversely scales with SDR
            seller_prices = [G_seller_buyer.nodes[neighbor]['obj'].price for neighbor in G_seller_buyer.neighbors(seller_id)]
            if seller_prices:
                min_seller_price = min(seller_prices)
                node.bid = min_seller_price / SDR  # Buyers bid inversely proportional to SDR

# Create a random network of buyers and sellers
def create_random_market_network(buyers, sellers):
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

# Main simulation function
def main_sdr_simulation():
    config = {
        "num_buyers": 5,
        "num_sellers": 3,
        "mu": 1.0,  # Long-term mean SDR
        "theta": 0.1,  # Speed of reversion to the mean
        "sigma": 0.2,  # Volatility of SDR
        "iterations": 100,
        "dt": 1.0,  # Time increment
    }

    # Initialize buyers and sellers
    sellers = [Node(f"Seller_{i}", price=np.random.uniform(50, 100)) for i in range(config["num_sellers"])]
    buyers = [Node(f"Buyer_{i}", price=0) for i in range(config["num_buyers"])]

    # Create a random market network
    G_seller_buyer = create_random_market_network(buyers, sellers)

    # Initialize SDR
    SDR = np.random.uniform(0.5, 1.5)
    SDR_history = [SDR]

    # Price and bid history
    price_history = {seller.id: [] for seller in sellers}
    buyer_price_history = {buyer.id: [] for buyer in buyers}

    # Simulation loop
    for iteration in range(config["iterations"]):
        # Update SDR using the OU process
        SDR = ou_process_sdr(SDR, mu=config["mu"], theta=config["theta"], sigma=config["sigma"], dt=config["dt"])
        SDR_history.append(SDR)

        # Update prices and bids based on SDR
        update_prices_and_bids_with_sdr(G_seller_buyer, SDR)

        # Track price and bid history
        for seller in sellers:
            price_history[seller.id].append(seller.price)
        for buyer in buyers:
            buyer_price_history[buyer.id].append(buyer.bid)

    # Visualization
    # Plot SDR evolution
    plt.figure(figsize=(10, 6))
    plt.plot(SDR_history, label="SDR")
    plt.axhline(y=1.0, color='r', linestyle='--', label="Equilibrium SDR")
    plt.xlabel("Iterations")
    plt.ylabel("Supply-Demand Ratio")
    plt.title("SDR Evolution Over Time")
    plt.legend()
    plt.show()

    # Plot price convergence
    plt.figure(figsize=(10, 6))
    for seller_id, prices in price_history.items():
        plt.plot(prices, label=f"Seller {seller_id}")
    plt.xlabel("Iterations")
    plt.ylabel("Price")
    plt.title("Seller Price Convergence Over Time")
    plt.legend()
    plt.show()

    # Plot bid convergence
    plt.figure(figsize=(10, 6))
    for buyer_id, bids in buyer_price_history.items():
        plt.plot(bids, label=f"Buyer {buyer_id}")
    plt.xlabel("Iterations")
    plt.ylabel("Bid")
    plt.title("Buyer Bid Convergence Over Time")
    plt.legend()
    plt.show()

main_sdr_simulation()
