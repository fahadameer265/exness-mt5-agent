"""Exness MT5 research and demo execution agent.

The package trains a directional probability model, then sends orders only
when the multi-timeframe rules and the risk gates agree. It does not claim
a live win rate. A 1:2 payoff can be profitable well below a 50% hit rate,
and a 70% model probability is a filter, not a promise.
"""

__version__ = "1.0.0"
