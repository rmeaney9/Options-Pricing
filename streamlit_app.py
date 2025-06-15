# bs_utils.py
import numpy as np
from scipy.stats import norm

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


# streamlit_app.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from bs_utils import call_price

# Streamlit App
st.title("Black-Scholes Option Price Heatmap")

option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
K = st.sidebar.number_input("Strike Price (K)", min_value=0.1, value=100.0, step=1.0, format="%.2f")
T_months = st.sidebar.number_input("Time to Maturity (Months)", min_value=1.0, value=12.0,step = 1.0,format="%2f")
T = T_months / 12

r = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0,value=0.05,step=0.01,format="%.2f")
entry_price = st.sidebar.number_input("Entry Option Price ($)", min_value = 0.0,value=5.0, step = 0.5, format="%.2f")

contract_size = st.sidebar.number_input("Number of Options", value=1, min_value=1)

position = st.sidebar.selectbox("Position", ["Long", "Short"])

S_min, S_max = st.sidebar.slider("Spot Price Range ($)", 20, 300, (50, 150))
S_vals = np.arange(S_min, S_max + 1, 1)
sigma_vals = np.linspace(0.05, 0.5, 100)
S_grid, sigma_grid = np.meshgrid(S_vals, sigma_vals)

# Compute PnL grid using vectorised implementation
price_grid = call_price(S_grid, K, T, r, sigma_grid, option_type)
if position == "Long":
    pnl_grid = (price_grid - entry_price) * contract_size
else:
    pnl_grid = (entry_price - price_grid) * contract_size

# Plot PnL heatmap

cmap = LinearSegmentedColormap.from_list("custom_pnl", ["red", "lightgray", "green"])
vmin = np.min(pnl_grid)
vmax = np.max(pnl_grid)

if vmin < 0 and vmax > 0:
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
else:
    norm = None  # fallback to linear scale if all PnL is positive or negative

fig, ax = plt.subplots(figsize=(8, 6))
c = ax.contourf(S_grid, sigma_grid, pnl_grid, levels=20, cmap=cmap, norm=norm)
fig.colorbar(c, ax=ax, label="PnL ($)")
ax.set_xlabel("Spot Price S ($)")
ax.set_ylabel("Volatility σ (Annual)")
ax.set_title(f"{position} {option_type.title()} Option PnL Heatmap")

st.pyplot(fig)