# Poisson-Astar

Poisson-A*: Fast and Safe Batch Path Planning via Edge-Weight Reshaping with a Global Poisson Field

## Repository Contents

| File | Description |
| ---- | ----------- |
| `map_extractor.py` | Extracts `map.json` from the figures in the `Figs/` folder. |
| `poisson_astar.py` | Main planner script. |
| `test_pipeline_.py` | Integrates the driver and visualization functions. After extracting the maps with `map_extractor.py`, use this script to test the full Poisson-A* pipeline. |
| `verify_field_heuristic.py` | Explains the negative certification of the Poisson field as a heuristic function. |

Note. poisson_xxx.npz is the cache of Poisson field, should lie in the .uep_cache folder. 
