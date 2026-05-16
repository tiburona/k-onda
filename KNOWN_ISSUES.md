## IO 

In addition to all input types that aren't supported yet, there's the class called NEVMixin only really reads from NEV files that were already translated to matfiles.

This is more of a question than an issue, and probably not an answerable one until I see mor examples, but I had a TODO that read "Should I eventually try to unify the helpers/properties of PhyOutput and Generic Spike Source?"  

## Model and Config

Experiment inherits from AnnotatorMixin, but is not currently setting an annotation upon adding an animal.

## Graph and Signals

That graph nodes and signals are synonymous is persistently confusing. There needs to be a refactor in which Signal is renamed to Node and Signal either inherits Node or  Signal becomes a Dataclass attached to Node with truly Signal-specific attributes, like sampling rate. 

To think about: do I need new `AxisKind`s for the dim over which signals are stacked?  For the integer indexes created by the user?

A `SignalStack` can't currently be compiled.  You probably mostly wouldn't want to, but there's no reason it shouldn't have the capability.

I had `PointProcessSignal` stop inheriting from DatasetSignal because a `PointProcessSignal` doesn't *have* to be a DatasetSignal`. However some PPS's are DatasetSignals and I should come up with an inheritance model that expresses it.  

Right now `payload` is guaranteed to return a signal of the same typ for DatasetSignals and I don't think that's right.

## Loci

I should add an 'absolute' option to `generate_markers`.

In loci/core.py `generate_markers` I suspect that the line that takes q.magnitude is going to break on the iterables (`offsets`, `positions`).

`Locus`/`LocusSet` has currently has no provenance record.  This is okay for config-derived epochs/events, but insufficient for future loci derived from signals.

There are some forbidden names for loci conditions (e.g., 'time', 'frequency'). Right not the code raises ValueError but when I actually have docs the messages should be replaced with a nice link to the docs.


## Transformers

I should go through all the calculators and make sure that rather than have generic config dictionaries they have separate arguments for the sake of inspectable signatures (or alternatively a params DataClass that would be inspectable), but I should also give them a generic config dictionary or some kind of callable that sweeps up all their params into wone object for the sake of pretty printing the graph.

I'm missing dispatch over `SignalMap` for Transformer and the select mixin.

`Spectrogram` currently assumes it's computed over time; that should be generalized.

`Spectrogram` currently tests `if isinstance(n_cycles, np.ndarray):`. This is only working because the only way anyone has run spectram is after defining config variables in Python; it will break as soon as config comes from a string source -- need to test for iterables more generally.

The only filter currently supported is sos; this needs to be expanded.

`Histogram` currently accepts a string `range_source`, "session", but there should be at least one other (which could maybe supercede "session") -- the smallest enclosing container on the histogram dim.  (e.g. if user has created epochs and the histogram is over time, "epochs".)

## Selection

`DimBounds` is written to such that it could have multiple dims, but locus and the selector logic are not.  Multiple dim select should be restored. 

You can't `select_point_process` yet because there's not yet support for ragged arrays.

The only kind of condition you can do is by equality; needs expansion.