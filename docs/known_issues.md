## IO 

In addition to all input types that aren't supported yet, the class called NEVMixin only really reads from NEV files that were already translated to matfiles.

This is more of a question than an issue, and probably not an answerable one until I see more examples, but I had a TODO that read "Should I eventually try to unify the helpers/properties of PhyOutput and Generic Spike Source?"  


## Model and Config

Experiment inherits from AnnotatorMixin, but is not currently setting an annotation upon adding a subject.


## Graph and Signals

That graph nodes and signals are synonymous is persistently confusing. There needs to be a refactor in which Signal is renamed to Node and Signal either inherits Node or  Signal becomes a Dataclass attached to Node with truly Signal-specific attributes, like sampling rate. 

To think about: do I need new `AxisKind`s for the dim over which signals are stacked?  For the integer indexes created by the user?

A `SignalStack` can't currently be compiled.  You probably mostly wouldn't want to, but there's no reason it shouldn't have the capability.

I had `PointProcessSignal` stop inheriting from DatasetSignal because a `PointProcessSignal` doesn't *have* to be a DatasetSignal`. However some PPS's are DatasetSignals and I should come up with an inheritance model that expresses it.  

Right now `payload` is guaranteed to return a signal of the same type for `DatasetSignal`s and I don't think that's right.

`Intersection` and `ApplyMask`'s `__call__`s need to be evaluated for how they're working with keys, how they apply to stacks, and in general to be brought up-to-date with the code base.  


## Loci

`Locus`/`LocusSet` has currently has no provenance record.  This is okay for config-derived epochs/events, but insufficient for future loci derived from signals.

There are some forbidden names for loci conditions (e.g., 'time', 'frequency'). Right now the code raises ValueError but when I actually have docs the messages should be replaced with a nice link to the docs.


## Transformers

I should go through all the calculators and make sure that rather than have generic config dictionaries they have separate arguments for the sake of inspectable signatures (or alternatively a params DataClass that would be inspectable), but I should also give them a generic config dictionary or some kind of callable that sweeps up all their params into one object for the sake of pretty printing the graph.

I'm missing dispatch over `SignalMap` for `Transformer` and the `select` mixin.

`Spectrogram` currently assumes it's computed over time; that should be generalized.

`Spectrogram` currently tests `if isinstance(n_cycles, np.ndarray):`. This is only working because the only way anyone has run spectrogram is after defining config variables in Python; it will break as soon as config comes from a string source -- need to test for iterables more generally.

The only filter currently supported is sos; this needs to be expanded.

`Histogram` currently accepts a string `range_source`, "session", but there should be at least one other (which could maybe supercede "session") -- the smallest enclosing container on the histogram dim.  (e.g. if user has created epochs and the histogram is over time, "epochs".)

Check the behavior of the various key modes (particularly "rename") and make sure the names here aren't misleading.


## Selection

`DimBounds` is written such that it could have multiple dims, but loci and the selector logic are not.  Multiple dim select should be restored. 

You can't `select_point_process` yet because there's not yet support for ragged arrays.

The only kind of filtering by condition you can do is by equality; needs expansion.

SpecifySelection transformers should probably be edited out of the graph after Slicer placement.

When making a new ordinal dim during selection, the program should validate and raise if the new loci belong to more than one earlier ordinal dim.  

In a true DAG (i.e., not a tree, with consumers that share an upstream node), `walk_graph` will create multiple SelectionSlicers.  At some point it's consumer named argument needs to be more expressive to prevent this kind of duplication.

`attach_condition_coords` only attaches coords if all loci have the condition (`conditions = reduce(and_, [set(l.conditions.keys()) for l in self.locus])`).  Should decide if that's the desired behavior.

Selection (or maybe aggregation, or both) is already slow.  It needs to be time profiled.


## Aggregation 

Right now, if you grouped the long axis (created by AssembleArray), you can't carry over any ungrouped coords.  For example, if you had neuron and neuron type on the long axis, and then you group by neurons, neuron_type is lost Eventually you should be able to migrate that coord over to the new axis, but that will require that somewhere knowledge is encoded about how to migrate them.

You should be able to calculate a simultaneous mean (i.e. unweighted by number of members of a group.)

