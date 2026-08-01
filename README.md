# SVI Volatility Smile Calibration for SPX Options

## Overview

This project calibrates the raw Stochastic Volatility Inspired (SVI) model to Bloomberg SPX option implied-volatility data. The objective is to fit the observed volatility smile while using ridge regularization to stabilize SVI parameters across consecutive market snapshots.

The dataset contains 42,160 valid option observations across 103 daily snapshots for the December 18, 2026 expiry.

## Methodology

1. Matched SPX options with the ESZ26 E-mini S&P 500 futures contract.
2. Adjusted the futures-based forward using near-ATM call–put implied-volatility differences.
3. Removed invalid implied volatilities, ITM-side observations, and extreme log-moneyness values.
4. Converted implied volatility to total variance:
   
   \[
   w = \sigma_{\text{imp}}^2 T
   \]

5. Calibrated the raw SVI model:

   \[
   w(k)=a+b\left[\rho(k-m)+\sqrt{(k-m)^2+\sigma^2}\right]
   \]

6. Used vega-squared weights and multi-start nonlinear optimization.
7. Jointly fitted all snapshots with a ridge penalty on standardized parameter changes.
8. Tested ridge strengths of 0, 0.01, 0.1, 1, and 10.

## Key Results

A ridge penalty of **λ = 0.1** was selected using a rule that allows no more than a 2% increase in weighted calibration RMSE.

| Model | Weighted RMSE | Mean scaled parameter step |
|---|---:|---:|
| Unpenalized | 0.2292 vol points | 0.2849 |
| Ridge, λ = 0.1 | 0.2313 vol points | 0.0469 |

The regularized model increased weighted RMSE by only **0.94%**, while reducing average standardized parameter movement by approximately **6.07×**.

The fitted SVI curves closely match the central and moderately out-of-the-money regions of the volatility smile. Larger residuals remain in the extreme wings, which receive lower weight because of their lower vegas.

## Outputs

- Cleaned SVI calibration dataset
- Baseline and ridge-regularized SVI parameters
- Lambda-sweep comparison table
- Fitted volatility-smile charts
- SVI parameter path charts
- Accuracy–stability trade-off chart

## Limitations

- The current dataset contains one option expiry, so the analysis represents a time series of volatility smiles rather than a complete multi-expiry surface.
- Bid and ask implied-volatility fields were unavailable, so the bid-ask violation penalty was not implemented.
- Extreme-wing observations are fitted less closely than central, higher-vega strikes.
