## IO

- It would be nice if file opening were lazy.  Right now, for instance, you load every animal's LFP files, maybe even preprocess them, regardless of whether you're going to use them in the analysis you're about to run.

## Plotting

- Just because the representation of the plotting partitions in code is nested, doesn't mean the spec has to be.  
Given that every partition can only have one child partition, it would be fine for the user to represent them as a flat list.
