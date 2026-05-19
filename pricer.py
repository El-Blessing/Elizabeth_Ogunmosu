import numpy as np

def price_barrier_option(S0, K, B, T, r, sigma, n, barrier_type='up'):
    """
    Cox-Ross-Rubinstein binomial pricer with discrete barrier monitoring.
    Matches Section 3.4-3.6 of the main project.
    """
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    p = np.clip(p, 0.0, 1.0)
    disc = np.exp(-r * dt)

    # Terminal stock prices & European call payoff
    j = np.arange(n + 1)
    S_T = S0 * (u ** j) * (d ** (n - j))
    V = np.maximum(S_T - K, 0.0)

    # Apply barrier condition at maturity
    if barrier_type == 'up':
        V[S_T >= B] = 0.0
    else:
        V[S_T <= B] = 0.0

    # Backward induction with discrete barrier check at each step
    for i in range(n - 1, -1, -1):
        j = np.arange(i + 1)
        S_i = S0 * (u ** j) * (d ** (i - j))
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])

        if barrier_type == 'up':
            V[S_i >= B] = 0.0
        else:
            V[S_i <= B] = 0.0

    return V[0]
