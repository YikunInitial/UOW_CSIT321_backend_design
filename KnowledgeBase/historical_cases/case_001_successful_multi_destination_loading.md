# Historical Case 001: Successful Multi-Destination Loading

This case involved steel coils for multiple destinations.
The successful strategy was to first group products by destination and then apply weight-aware greedy placement.

Important observations:
- Destination grouping improved unloading clarity.
- Strict destination grouping alone was not enough.
- Heavy coils were placed closer to the centerline.
- After greedy placement, balance checks were applied.
- The final solution reduced the number of wagons while satisfying boundary and overlap constraints.

Recommended strategy:
Use destination grouping first, then apply balance-aware greedy placement.