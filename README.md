# Mapinator Simulation
Simulation of Mapinator Reverse Directed Search Model

Based on Peters (2021): https://montoya.econ.ubc.ca/papers/markets/markets.pdf

Simluation produces an adjacency matrix of unversity-PhD garduate matches, and uses maximum likelihood estimation to estimate the structural parameters of the offer distribution.

Written in Python. File runs in Jupyter Notebook.

===========

### To use the simulation, use the following function:

`mapinator_nxm(uni_and_tier:bool,
                  universities:int,
                  graduates:int,
                  tier_probs:List[float],
                  tiervalues:List[float],
                  rounds:int,
                  stdevs:List[float],
                  means=List[float])`

Where the researcher may choose the following parameters to simluate a reverse directed search game:

1. `uni_and_tier`: True if you want to produce a tier and university adjacency matrix. False if you want to produce only the tier adjacency matrix. Note that the university adjacency matrix is computationally demanding at high dimensions.

2. `universities`: Number of universities competing for graduates.

3. `graduates`: Number of PhD graduates on the market.

4. `tier_probs`: List of probabilities that any given university is from a tier. Assumes 4 tiers, 1 being the most prestigious, 4 being the least.

5. `tiervalues`: List of values of graduates from each tier. Intended for values between 0 and 1.

6. `rounds`: Number of times simulation is repeated. Adjacency matrix cells are summed over rounds.

7. `stdevs`: Standard deviations of the truncated normal offer distribution for each tier. Ex. [0.1, 0.1, 0.2, 0.2] gives the standard deviations for tier 1, tier2, tier3, and tier4 offers.

8. `means`: Averages of the truncated normal offer distribution for each tier. Ex. [0.8, 0.7, 0.6, 0.5] gives the means for tier 1, tier2, tier3, and tier4 offers.

