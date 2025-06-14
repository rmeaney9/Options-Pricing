This project uses the **Black–Scholes analytical formula** to compute European option prices:

$$
\begin{aligned}
C &= S \cdot N(d_1) - K e^{-rT} N(d_2) \\
P &= K e^{-rT} N(-d_2) - S N(-d_1)
\end{aligned}
$$

Where:

- $S$ = Spot price  
- $K$ = Strike price  
- $T$ = Time to maturity (in years)  
- $r$ = Risk-free interest rate  
- $\sigma$ = Volatility (annualised)  
- $N(\cdot)$ = Standard normal cumulative distribution function (CDF)

---
The app calculates and visualizes **profit and loss (PnL)** as a function of spot price and volatility:

$$
\text{PnL} = (\text{Option Price} - \text{Entry Price}) \times \text{Contracts}
$$
