# Black-Scholes Options PnL Visualizer

An interactive Streamlit app for visualizing the profit and loss (PnL) profile of European call and put options using the Black–Scholes analytical model.

---

## Overview

This tool lets users explore how option PnL varies with changes in spot price and implied volatility. It calculates theoretical option prices based on the Black–Scholes formula and compares them to a user-defined entry price to visualize potential outcomes across a 2D grid.

---

## Model Description

The theoretical price of a European option is calculated using the Black–Scholes formula:

$$
\begin{aligned}
C &= S \cdot N(d_1) - K e^{-rT} N(d_2) \\
P &= K e^{-rT} N(-d_2) - S N(-d_1)
\end{aligned}
$$

Where:

- \( S \): Spot price  
- \( K \): Strike price  
- \( T \): Time to maturity (in years)  
- \( r \): Risk-free interest rate  
- \( $$\sigma$$): Volatility of the underlying asset  
- \( N): Standard normal cumulative distribution function

With:

$$
\begin{aligned}
d_1 &= \frac{\ln(S/K) + (r + \frac{1}{2} \sigma^2) T}{\sigma \sqrt{T}} \\
d_2 &= d_1 - \sigma \sqrt{T}
\end{aligned}
$$

The model accounts for both **intrinsic value** and **time value**.

---

## PnL Calculation

After the user specifies an entry price and position (long or short), the app calculates:

$$
\text{PnL} = (\text{Option Price} - \text{Entry Price}) \times \text{Contracts}
$$

Color-coded output:
- Green for profitable positions
- Red for losing positions
- Neutral for breakeven

---

## Running the App

1. Clone the repository:
   ```bash
   git clone https://github.com/rmeaney9/options-pricing
   cd options-pricing
2. pip install -r requirements.txt
3. streamlit run streamlit_app.py


## ✍️ Author

Ronan Meaney
[LinkedIn](linkedin.com/in/ronan-meaney) | [GitHub](github.com/rmeaney9)
