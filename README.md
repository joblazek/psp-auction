# psp-auction

    Implements Algorithm 1 (Lazar & Semret, Appendix B),
    listens for seller updates.

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



