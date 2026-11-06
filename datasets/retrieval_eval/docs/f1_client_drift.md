# Client drift in federated averaging

Abstract

Federated averaging struggles when client data is not identically distributed. We study client drift as the number of local epochs grows.

Methods

One hundred simulated clients each held data from two of ten classes. We varied local epochs from one to twenty and tracked global accuracy.

Results

Client drift grew with the number of local epochs, and test accuracy after fifty rounds fell from 81 percent at one epoch to 63 percent at twenty. A small proximal term reduced the loss of accuracy.
