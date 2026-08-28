# Tests

Compare implementations against an FP32 gathered-KV reference. Record maximum
absolute error, maximum relative error, cosine similarity, and NaN/Inf status.

Do not relax tolerances without recording the baseline Triton error distribution.
