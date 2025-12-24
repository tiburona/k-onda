## Architecture

- My robot overlord threw some shade about my classes not being cooperative.  Evaluate whether a refactor is necessary.

- Idea from Sam -- possible refactor: move calculation methods off data objects to seperate calculator classes

- Right now if we record spike data from more than one brain region we can't represent that.  Thinking about this more, given the choices I've already made, it might make sense to reverse the hierarchical order of unit and period and have Animal be the only period constructor.  That has to be contemplated though.

## IO

- It would be nice if file opening were lazy. The only gating is by analysis type, but especially when testing you just want a subset of animals. (And on this front, maybe it would save some time to store pickles of the processed lfp data? And pickles of the results of getting the period onsets from NEV's?  This is all very annoyingly slow.)

- User might want to combine onset information from nev and other external spec.


## Plotting

- Just because the representation of the plotting partitions in code is nested, doesn't mean the spec has to be.  Given that every partition can only have one child partition, it would be fine for the user to represent them as a flat list.

- Short of a full-fledged custom GUI, what could interactive plots give the user? Should I have used Plotly instead of matplotlib?  Is it worth a refactor?


## Performance

- Multitaper calculations of coherence events are very slow, and would be faster if they were vectorized.  Event can be a public abstraction that's actually an index into periods or segments.  Should all events be represented the same way?  

- Caching needs to be reviewed. That method should be moved out of utils, perhaps into core.  It's applied very narrowly right now and some obvious uses for caching just never touch it.  It needs reviewing whether period type always works right in the "selected_period_types" case.  There's a depth switch which is apparently not being used right now and that should be reviewed too.   

- Would PSTH be faster if in the beginning of any operation, the list of spike times were converted into a spike train and find spikes became an indexing rather than a search operation?  Note: robot overlord doesn't like this idea, but suggests: "If events are short and numerous, precompute spike indices per period type once and reuse."








