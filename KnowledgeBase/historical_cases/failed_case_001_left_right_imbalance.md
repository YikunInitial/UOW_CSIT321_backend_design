# Failed Case 001: Left-Right Imbalance

This failed case placed most heavy products on one side of the wagon.
Although the total weight did not exceed the wagon capacity, the centerline balance failed.

Failure reason:
- Too many heavy coils were assigned to the same side.
- The algorithm prioritised compact placement but ignored left-right weight balance.
- The visualisation showed obvious asymmetry.

Lesson:
For heavy steel coils, compact placement should not be the only objective.
The optimisation agent must include balance-aware scoring or a post-placement validation step.