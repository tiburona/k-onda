## Architecture

- My robot overlord threw some shade about my classes not being cooperative.  Evaluate whether a refactor is necessary.

## IO

- It would be nice if file opening were lazy. The only gating is by analysis type, but especially when testing you just want a subset of animals. (And on this front, maybe it would save some time to store pickles of the processed lfp data? And pickles of the results of getting the period onsets from NEV's?  This is all very annoyingly slow.)


## Plotting

- Just because the representation of the plotting partitions in code is nested, doesn't mean the spec has to be.  Given that every partition can only have one child partition, it would be fine for the user to represent them as a flat list.

## Performance

- Multitaper calculations of coherence events are very slow, and would be faster if they were vectorized.  Event can be a public abstraction that's actually an index into periods or segments.  Should all events be represented the same way?  

- Caching needs to be reviewed. That method should be moved out of utils, perhaps into core.  It's applied very narrowly right now and some obvious uses for caching just never touch it.  It needs reviewing whether period type always works right in the "selected_period_types" case.  There's a depth switch which is apparently not being used right now and that should be reviewed too.     
