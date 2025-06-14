import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


# Black-Scholes helpers
def bs_inputs(S, K, T, r, sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return d1, d2

def call_price(S,K,T,r,sigma,option_type="call"):
    d1, d2 = bs_inputs(S,K,T,r,sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Streamlit App
st.title("Black-Scholes Option Price Heatmap")

option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
K = st.sidebar.slider("Strike Price (K)", 50, 150, 100)
T_months = st.sidebar.slider("Time to Maturity (T in months)", 1, 24, 12)
T = T_months/12
r = st.sidebar.slider("Risk-Free Rate (r)", 0.00, 0.20, 0.05)
entry_sigma = st.sidebar.slider("Entry Volatility (σ) for Premium", 0.05, 0.5, 0.2)

use_model_price = st.sidebar.checkbox("Use model option price as entry", value=True)
if use_model_price:
    entry_price = call_price(S=K, K=K, T=T, r=r, sigma=entry_sigma, option_type=option_type)
else:
    entry_price = st.sidebar.number_input("Entry Option Price ($)", value=10.0, min_value=0.0)

contract_size = st.sidebar.number_input("Number of Options", value=1, min_value=1)

position = st.sidebar.selectbox("Position", ["Long", "Short"])

S_vals = np.arange(50, 151, 1)
sigma_vals = np.linspace(0.05, 0.5, 100)
S_grid, sigma_grid = np.meshgrid(S_vals, sigma_vals)

# Compute PnL grid
pnl_grid = np.zeros_like(S_grid)
for i in range(S_grid.shape[0]):
    for j in range(S_grid.shape[1]):
        price = call_price(S_grid[i, j], K, T, r, sigma_grid[i, j], option_type)
        pnl = price - entry_price if position == "Long" else entry_price - price
        pnl_grid[i, j] = pnl * contract_size

# Plot PnL heatmap
cmap = LinearSegmentedColormap.from_list("custom_pnl", ["red", "lightgray", "green"])
norm = TwoSlopeNorm(vcenter=0, vmin=np.min(pnl_grid), vmax=np.max(pnl_grid))

fig, ax = plt.subplots(figsize=(8, 6))
c = ax.contourf(S_grid, sigma_grid, pnl_grid, levels=20, cmap=cmap, norm=norm)
fig.colorbar(c, ax=ax, label="PnL ($)")
ax.set_xlabel("Spot Price (S)")
ax.set_ylabel("Volatility (σ)")
ax.set_title(f"{position} {option_type.title()} Option PnL Heatmap")

st.pyplot(fig)