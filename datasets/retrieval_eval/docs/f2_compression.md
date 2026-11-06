# Communication compression for federated training

Abstract

Sending full model updates is expensive. We compare top-k sparsification and quantisation as ways to compress gradient updates.

Methods

Each round, clients sent either the full update, the largest one percent of entries, or an eight-bit quantised update.

Results

Top-k sparsification cut the bytes per round by 97 percent while accuracy stayed within one point of the full update. Eight-bit quantisation saved 75 percent of the bytes.
