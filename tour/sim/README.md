Trajectory Simulation

The states of the Markov chain (MC) corresponds to the categories of POIs, there is a special state "REST" which represents that people are having rests after some travelling.
 
Simulate trajectories using the transition matrix of the MC, when choosing a specific POI within a certain category, use the following rules:

1. The Nearest Neighbor of the current POI 
1. The most Popular POI 
1. A random POI choosing with probability proportional to the reciprocal of its distance to current POI 
1. A random POI choosing with probability proportional to its popularity
 
Run the simulation a number of times, compute the transition probabilities (transition matrix) using the generated observations:
```$ python3 simulation.py N``` 

where ```N``` is the number of simulating steps (or observations), e.g. ```100000```
