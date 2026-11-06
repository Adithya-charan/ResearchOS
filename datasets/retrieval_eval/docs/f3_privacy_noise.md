# Differential privacy noise in training rounds

Abstract

Adding calibrated noise limits what an update reveals about one participant. We measure the accuracy cost of gaussian noise with gradient clipping.

Methods

Updates were clipped to a fixed norm and gaussian noise was added to reach privacy budgets with epsilon between one and ten.

Results

At epsilon of ten the accuracy loss was 2 points, and at epsilon of one it was 14 points. Tighter clipping helped at small budgets.
