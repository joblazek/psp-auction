#Implementation of Algorithm 1

import time
import math
import numpy as np
from typing import Dict, Tuple, List
import pandas as pd

class VirtualAuctionServer:
    """A virtual server to hold bids and compute allocations."""
    def __init__(self, Q_max: float):
        """
        :param Q_max: total supply at this node
        :param s = {i: s_i}: Bid profile: (q_i, p_i) for agent i
        """
        self.s: Dict[str, Tuple[float, float]] = {}
        self.Q_max = Q_max

    def update_bid(self, i: str, s_i: Tuple[float, float]):
        """Update the bid profile s_i = (q_i, p_i) for agent i."""
        self.s[i] = s_i

    def get_s_hat_minus(self, i: str) -> Dict[str, Tuple[float, float]]:
        """Retrieve ŝ_{-i}: bids of all agents except i."""
        return {j: s_j for j, s_j in self.s.items() if j != i}

    def allocate_and_price(self) -> Tuple[str, float]:
        """
        Compute allocation a_i and payment c_i for a single-unit second-price auction.
        - Winner: a_i = argmax_j p_j
        - Payment: second-highest price
        Returns (winner_id, payment).
        """
        if not self.s:
            return None, 0.0
        sorted_by_price = sorted(self.s.items(), key=lambda x: x[1][1], reverse=True)
        winner, (_, highest_price) = sorted_by_price[0]
        pay_price = sorted_by_price[1][1] if len(sorted_by_price) > 1 else 0.0
        return winner, pay_price

class Agent:
    """
    Implements Algorithm 1 (Lazar & Semret, Appendix B).

    Initialization:
        s_i = 0
        ŝ_{-i} = ∅

    1. Compute truthful ε-best reply t_i = (v_i, w_i):
        v_i = [ sup G_i(ŝ_{-i}) – ε / θ_i′(0) ]_+
        w_i = θ_i′(v_i)

        where sup G_i(ŝ_{-i}) is
        sup { z ∈ [0, Q] :
              z ≤ Q_i(θ_i′(z), ŝ_{-i})
           and ∫₀ᶻ P_i(ζ, ŝ_{-i}) dζ ≤ b_i }

    2. If u_i(t_i, ŝ_{-i}) > u_i(s_i, ŝ_{-i}) + ε, then
           s_i ← t_i

    3. Sleep 1 second and repeat.
    """
    def __init__(self, i: str, epsilon: float, b_i: float, q_i: float, kappa_i: float, server: VirtualAuctionServer):
        """
        :param i: agent identifier
        :param epsilon: ε threshold
        :param b_i: budget b_i (max integral cost)
        :param q_i: physical capacity (max quantity) for player i
        :param kappa_i: valuation intensity κ_i > 0
        """
        self.i = i
        self.epsilon = epsilon
        self.b_i = b_i
        self.q_i = q_i
        self.kappa_i = kappa_i

        # Current bid s_i = (q_i, p_i); initially zero
        self.s_i: Tuple[float, float] = (0.0, 0.0)

        # Server reference
        self.server = server
        self.server.update_bid(self.i, self.s_i)

    def theta_i(self, z: float) -> float:
        """
        Valuation function θ_i(z).
        We are using the parabolic valuation function:
            θ_i(z) = (κ_i / 2) * (min(z, q_i))^2
                     + κ_i * q_i * (min(z, q_i))

        The valuation function is strictly concave for z≤qi​,
        ensuring diminishing marginal returns.
        Beyond qi​, the valuation function becomes flat,
        reflecting the physical limit of the resource.

        :param z: allocated quantity
        :returns: utility θ_i(z)
        """
        m = min(z, self.q_i)
        return 0.5 * self.kappa_i * m**2 + self.kappa_i * self.q_i * m

    def theta_i_prime(self, z: float) -> float:
        """
        Marginal valuation θ_i′(z).
        i.e., the derivative of the parabolic valuation function θ_i:

            θ_i(z) = (κ_i / 2) * (min(z, q_i))^2
                     + κ_i * q_i * min(z, q_i)

        ⇒

            θ_i'(z)
            = κ_i * (z + q_i),    for 0 ≤ z < q_i
            = 0,                  for z ≥ q_i
        """
        if z < self.q_i:
            return self.kappa_i * (z + self.q_i)
        else:
            return 0.0

    def sup_G(self, s_hat_minus: Dict[str, Tuple[float, float]]) -> float:
        """
        Feasible set uses Q_i at θ'(z): z ≤ Q_i(θ'(z); s_{-i}).
        """
        best_z = 0.0
        for z in np.linspace(0, self.Q_max, 100):
            price = self.theta_i_prime(z)
            if (z <= self.Q_i(price, s_hat_minus)
                    and self.integral_P(z, s_hat_minus) <= self.b_i):
                best_z = z
        return best_z

    def Q_i(self, p_i: float, s_hat_minus: Dict[str, Tuple[float, float]]) -> float:
        """
        Conditional raw supply curve Q̄_i(p_i; s_{-i}):

        Given your bid price p_i and the profile of opponents' bids s_{-i},
        this returns the maximum quantity available to you after fully
        serving all opponents whose bids strictly exceed p_i.

        Mathematically:
            Q̄_i(p_i; s_{-i}) = max { Q_max - ∑_{j: p_j > p_i} q_j, 0 }

        where each opponent j requests quantity q_j at price p_j.
        """
        # Start with full capacity
        remaining = self.server.Q_max
        # Subtract the quantities of all opponents
        # whose bid price p_j strictly exceeds p_i
        for (q_j, p_j) in s_hat_minus.values():
            if p_j > p_i:
                remaining -= q_j
        # Cannot go below zero
        return max(remaining, 0.0)

    def Q_i_bar(self, p_i: float, s_hat_minus: Dict[str, Tuple[float, float]]) -> float:
        """
        Conditional raw supply curve Q̄_i(p_i; s_{-i}):

        Given your bid price p_i and the profile of opponents' bids s_{-i},
        this returns the maximum quantity available to you after fully
        serving all opponents whose bids exceed or match p_i.

        Mathematically:
            Q̄_i(p_i; s_{-i}) = max { Q_max - ∑_{j: p_j >= p_i} q_j, 0 }

        where each opponent j requests quantity q_j at price p_j.
        """
        # Start with full capacity
        remaining = self.server.Q_max
        # Subtract the quantities of all opponents
        # whose bid price p_j strictly exceeds p_i
        for (q_j, p_j) in s_hat_minus.values():
            if p_j >= p_i:
                remaining -= q_j
        # Cannot go below zero
        return max(remaining, 0.0)

    def sup_G(self, s_hat_minus: Dict[str, Tuple[float, float]]) -> float:
        """
        Compute sup G_i(s_{-i}), where
            G_i(s_{-i}) = { z ∈ [0, Q_max] :
                z ≤ Q_i(θ'(z); s_{-i})
              and ∫₀ᶻ P_i(ζ; s_{-i}) dζ ≤ b_i }

        Notice that here we call Q_i at the price θ'(z):
            z ≤ Q_i(θ'(z); s_{-i}).
        """
        best_z = 0.0
        for z in np.linspace(0, self.q_i, 100):
            if (z <= self.Q_i(self.theta_i_prime(z), s_hat_minus) and self.integral_P(z, s_hat_minus) <= self.b_i):
                best_z = z
        return best_z


    def compute_t_i(self, s_hat_minus: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
        """
        Compute t_i = (v_i, w_i) per Proposition 1:

            v_i = [sup_G - ε/θ_i′(0)]_+
            w_i = θ_i′(v_i)
        """
        G_sup = self.sup_G(s_hat_minus)
        adjustment = self.epsilon / self.theta_i_prime(0.0)
        v_i = max(G_sup - adjustment, 0.0)
        w_i = self.theta_i_prime(v_i)
        return v_i, w_i

    def a_i(self, s_i: Tuple[float, float], s_hat_minus: Dict[str, Tuple[float, float]]) -> float:
        """
        Allocation rule:
            a_i(s) = q_i ∧ Q_i(p_i, ŝ_{-i})
        """
        q_i, p_i = s_i
        return min(q_i, self.Q_i_bar(p_i, s_hat_minus))

    def P_i(self, z: float, s_hat_minus: Dict[str, Tuple[float, float]]) -> float:
        """
        Price density function:

            P_i(z; ŝ_{-i}) = inf { y ≥ 0 : Q_i(y; ŝ_{-i}) ≥ z }.

        We sample candidate prices (0 and all opponents' p_j),
        and return the smallest y meeting Q_i(y) ≥ z.
        """
        candidates = sorted({0.0} | {p for (_, p) in s_hat_minus.values()})
        # Find the smallest y that yields at least z
        for y in candidates:
            if self.Q_i(y, s_hat_minus) >= z:
                return y
        return float('inf')


    def integral_P(self, z: float, s_hat_minus: Dict[str, Tuple[float, float]]) -> float:
        """Compute ∫₀ᶻ P_i(ζ, ŝ_{-i}) dζ via simple trapezoidal rule.
                where:
          - z = allocation to i under PSP (infinitely divisible),
          - P_i(z; ŝ_{-i}) = inf{y ≥ 0 : Q_i(y; ŝ_{-i}) ≥ z}.
        """

        N = 100  # increase resolution for accuracy
        zs = np.linspace(0, z, N + 1)
        Ps = [self.P_i(z, s_hat_minus) for z in zs]
        # trapezoidal rule: sum((P[k] + P[k+1]) / 2 * dz)
        dz = z / N
        total = sum((Ps[k] + Ps[k+1]) * 0.5 for k in range(N)) * dz
        return total

    def c_i(self, s_i: Tuple[float, float], s_hat_minus: Dict[str, Tuple[float, float]]) -> float:
        """
        Cost under progressive allocation:

            c_i(s) = ∫₀^{a_i(s)} P_i(z; ŝ_{-i}) dz

        where:
          - a_i(s) = allocation to i under PSP (infinitely divisible),
          - P_i(z; ŝ_{-i}) = inf{y ≥ 0 : Q_i(y; ŝ_{-i}) ≥ z}.

        We compute this via a trapezoidal rule over N steps.
        """
        # Determine allocation
        a = self.a_i(s_i, s_hat_minus)
        # Numerically integrate P_i from 0 to a
        return self.integral_P(a, s_hat_minus)

    def u_i(self, s_i: Tuple[float, float], s_hat_minus: Dict[str, Tuple[float, float]]) -> float:
        """
        Utility:
            u_i(s) = θ_i(a_i(s)) − c_i(s)
        """
        a = self.a_i(s_i, s_hat_minus)
        return self.theta_i(a) - self.c_i(s_i, s_hat_minus)

    def step(self) -> bool:
        """
        Perform one Algorithm 1 iteration.
        Returns True if the agent updated its bid (s_i changed), False otherwise.
        """
        s_hat_minus = self.server.get_s_hat_minus(self.i)
        t_i = self.compute_t_i(s_hat_minus)
        old_s_i = self.s_i

        # Check utility improvement condition
        if self.u_i(t_i, s_hat_minus) > self.u_i(old_s_i, s_hat_minus) + self.epsilon:
            # Update bid
            self.s_i = t_i
            self.server.update_bid(self.i, self.s_i)
            return True
        return False

def print_round_info(round_num: int,
                     agents: List[Agent],
                     server: VirtualAuctionServer) -> None:
    """
    Displays a table for round `round_num` showing, per agent:
      • q_i, p_i
      • a_i(s), c_i(s), u_i(s)
    Then a summary row for the winner and what they pay.
    """
    records = []
    # 1) collect each agent's data
    for ag in agents:
        s_hat = server.get_s_hat_minus(ag.i)
        q_i, p_i = ag.s_i
        records.append({
            "Agent":   ag.i,
            "q_i":     f"{q_i:.2f}",
            "p_i":     f"{p_i:.2f}",
            "alloc":   f"{ag.a_i(ag.s_i, s_hat):.2f}",
            "cost":    f"{ag.c_i(ag.s_i, s_hat):.2f}",
            "utility": f"{ag.u_i(ag.s_i, s_hat):.2f}",
        })

    # 2) get winner & payment
    raw = server.allocate_and_price()
    winner, raw_pay = raw if isinstance(raw, tuple) else (None, raw)
    # if allocate_and_price accidentally returns (q,p) for price:
    if isinstance(raw_pay, tuple) and len(raw_pay) == 2:
        pay_price = raw_pay[1]
    else:
        pay_price = float(raw_pay)

    # 3) append summary row
    records.append({
        "Agent":   "",
        "q_i":     "",
        "p_i":     "",
        "alloc":   "",
        "cost":    "",
        "utility": ""
    })

    # 4) render
    df = pd.DataFrame(records)
    print(f"### Round {round_num}")
    print(df)

def run_simulation(agents: List, server, rounds: int = 10, delay: float = 1.0) -> None:
    """
    Runs the auction for up to `rounds`, stopping if no bids change in a round.
    Prints each round via pandas DataFrames.
    """
    for t in range(1, rounds + 1):
        changed_any = False

        # 1) Each agent takes one step
        for agent in agents:
            if agent.step():
                changed_any = True

        # 2) Check for convergence
        if not changed_any:
            # Final note
            print("Auction Converged")
            break

        # 3) Print round summary
        print_round_info(t, agents, server)

        # 4) Pause before next round
        time.sleep(delay)


# Example usage
if __name__ == "__main__":

    server = VirtualAuctionServer(Q_max=5.0)
    # Instantiate agents with positional args:
    # (i, epsilon, b_i, Q_max, q_i, kappa_i, server)
    agents: List[Agent] = [
        Agent("A", 0.1, 10.0, 3.0, 2.0, server),
        Agent("B", 0.1, 12.0, 1.5, 1.8, server),
        Agent("C", 0.1, 15.0, 2.8, 1.5, server),
    ]

    print("## Auction Simulation")
    run_simulation(agents, server, rounds=10, delay=0.5)

