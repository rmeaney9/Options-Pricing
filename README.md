This project uses the **Black–Scholes analytical formula** to compute European option prices:

<div align="center">

\[
\begin{aligned}
C &= S \cdot N(d_1) - K e^{-rT} N(d_2) \\
P &= K e^{-rT} N(-d_2) - S N(-d_1)
\end{aligned}
\]

</div>

Where:

- \( S \) = Spot price  
- \( K \) = Strike price  
- \( T \) = Time to maturity (in years)  
- \( r \) = Risk-free interest rate  
- \( \sigma \) = Volatility (annualised)  
- \( N(\cdot) \) = Standard normal cumulative distribution function (CDF)

### 🔍 What the App Visualises

The app computes **PnL (Profit and Loss)** across a range of spot prices and volatilities using:

<div align="center">

\[
\text{PnL} = (\text{Option Price} - \text{Entry Price}) \times \text{Contracts}
\]

</div>
