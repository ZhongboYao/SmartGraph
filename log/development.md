# Baseline
#### Pipeline
read pdf -> extract topics using LLM API -> merge with the existing one by key matching.
#### Current problem:
Key matching is too strict, should consider semantic meanings

# V1
#### Pipeline
Improve merging with embedding similarities from sentence-transformers and indexed by FAISS.
It is faster and cheaper than LLMs.

#### Result (precision):
True positive: 17.3%
Hard negatives: 100%
Soft negatives: 99%

#### Problem Pattern:
Fail to merge abbrevations, like BPE <-> Byte Positional Encoding.
Fail to merge when two items with different number of words but the same meaning, like Transformer VS Transformer Architecture.
The good news here is it wont merge the ones shouldnt be merged, but it will ignore a lot of nodes that should be merged, which can be correctified manually.

### Adjustment
Ask LLM to generate full names to avoid failures due to abbrevations

# V2
#### Pipeline
Introduce a function that enables users to switch among graphs, rename graphs and create/delete graphs.