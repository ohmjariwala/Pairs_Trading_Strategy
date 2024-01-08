# Pairs_Trading_Strategy

This code implements a pairs trading strategy and employs statistical methods and regression analysis to identify pairs of stocks that exhibit cointegration- a statistical property that indicates a relationship between the price of two securities. 

The cointegration is explored by performing an Augmented Dickey-Fuller test to detect cointegration between various stocks from the technology sector (could be changed by simply modifying stock list)


# Pairs Trading Strategy Execution:
Utilizing the most volatile and most cointegrated ones, the script develops a pairs trading strategy. It continuously monitors the spread between their prices and executes buy and sell signals

The strategy employs statistical arbitrage, taking advantage of temporary mispricing between the pair. Trades are initiated when the z-score exceed a predefined threshold.

The implementation manages trade sizes based on a percentage of the initial capital and tracks the profit and loss (P&L) over time.

Finally, the script plots the P&L curve, showcasing the strategy's profitability and performance over the trading period.

Competencies Displayed:
# Statistical analysis and hypothesis testing
# Application of quantitative techniques
# Risk management and controlled trade execution
# Strategic analysis and decision-making
