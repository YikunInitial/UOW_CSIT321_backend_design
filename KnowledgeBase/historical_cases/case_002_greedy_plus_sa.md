# Historical Case 002: Greedy Plus Simulated Annealing

In this case, the greedy algorithm generated a quick initial loading solution.
However, the initial solution had several balance issues.

A simulated annealing improvement step was then applied.
The improvement process swapped product positions and adjusted orientations to reduce balance violations.

Outcome:
- The number of wagons remained the same.
- Centerline balance improved.
- Diagonal balance improved.
- The final visualisation was easier for operators to verify.

Recommended strategy:
Use greedy placement for the initial solution and simulated annealing for improvement.