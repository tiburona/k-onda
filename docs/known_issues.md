## IO 

In addition to all input types that aren't supported yet, the class called NEVMixin only really reads from NEV files that were already translated to matfiles.

This is more of a question than an issue, and probably not an answerable one until I see more examples, but I had a TODO that read "Should I eventually try to unify the helpers/properties of PhyOutput and Generic Spike Source?"  


## Model and Config

Experiment inherits from AnnotatorMixin, but is not currently setting an annotation upon adding a subject.


## Coordinates and schemas

Right now the only way to express tolerance for floating point differences across a number of calculators is 
with the parameter tolerance_decimals. This is unsatisfactory -- the meaning of a decimal changes with the unit.
I need some kind of general, unit-aware approach to this.  

Arithmetic calculators compute which coordinates are non-matching and appropriately drop them,
but any calculator that can take more than one input needs to do the same. (Right now this is basically
Intersection and ApplyMask, but this is a foundational issue that should be addressed for all multi-input 
transformers.)  Also right now absolute coordinates are automatically assumed to differ
in a multi input operation, but that's not always so.  For example, you could have decide you wanted to 
subtract delta from theta over the same time range.  I need a way to infer that the absolute time
coordinates are safe (probably using the provenance info of the signal to figure out that it came 
from the same source.) 

`Intersection` does not yet support signals sampled on genuinely different grids
(for example, a 30 fps video-derived mask and power calculated every 0.01 seconds).
It needs an alignment/resampling policy and a decision about the output
grid. `Intersection` should also support locations described by multiple paired
coordinates (for example, repeated observations with `x` and `y` coordinates).  This is probably a 
broader issue, and is another one of those things that needs to be centrally addressed for all multi
unit calculators.  


## Graph and Signals

That graph nodes and signals are synonymous is persistently confusing. There needs to be a refactor in which Signal is renamed to Node and Signal either inherits Node or  Signal becomes a Dataclass attached to Node with truly Signal-specific attributes, like sampling rate. 

To think about: do I need new `AxisKind`s for the dim over which signals are stacked?  

A `SignalStack` can't currently be compiled.  You probably mostly wouldn't want to, but there's no reason it shouldn't have the capability.

I had `PointProcessSignal` stop inheriting from DatasetSignal because a `PointProcessSignal` doesn't *have* to be a DatasetSignal`. However some PPS's are DatasetSignals and I should come up with an inheritance model that expresses it.  

Right now `payload` is guaranteed to return a signal of the same type for `DatasetSignal`s and I don't think that's right.


## Loci

`Locus`/`LocusSet` currently has no provenance record.  This is okay for config-derived epochs/events, but insufficient for future loci derived from signals.

There are some forbidden names for loci conditions (e.g., 'time', 'frequency'). Right now the code raises ValueError but when I actually have docs the messages should be replaced with a nice link to the docs.


## Transformers

I'm missing dispatch over `SignalMap` for `Transformer` and the `select` mixin.

`Spectrogram` currently assumes it's computed over time; that should be generalized.

The only filter currently supported is sos; this needs to be expanded.

`MedianFilter` currently delegates to `scipy.signal.medfilt`, which treats values
outside ordinary array boundaries as zero. This can affect filtered values near
either edge even when the kernel is no larger than the dimension. K-Onda needs
to choose an explicit boundary policy and decide whether that policy should be
fixed or user-configurable.

`Histogram` currently accepts a string `range_source`, "session", but there should be at least one other (which could maybe supercede "session") -- the smallest enclosing container on the histogram dim.  (e.g. if user has created epochs and the histogram is over time, "epochs".)

Check the behavior of the various key modes (particularly "rename") and make sure the names here aren't misleading.

`Intersection` and `ApplyMask`'s `__call__`s need to be evaluated for how they're working with keys, how they apply to stacks, and in general to be brought up-to-date with the code base.  


## Selection

`select()` needs validation. For most other fluent methods, the policy was to validate in the transformer,
but select normalizes user input before passing object types the fluent method user won't recognize to
the transformer, so it needs validation that reflect the inputs the user passes, while the transformer
needs validation that reflect its configured inputs.

Disjoint time selection doesn't provide an observation duration for downstream calculations. `SliceSelection.start_and_duration()` stores tuples of per-bound starts and durations, but `Rate` requires one scalar denominator.

Selecting a `LocusSet` from continuous data without `new_dim` is not yet
implemented. The operation needs a policy for representing the union
of disjoint or overlapping regions on the original axis, including whether to
drop or mask gaps. It also needs to prevent downstream calculations from assuming
it is regularly sampled.

`DimBounds` is written such that it could have multiple dims, but loci and the selector logic are not.  Multiple dim select should be restored. 

You can't `select_point_process` yet because there's not yet support for ragged arrays.

The only kind of filtering by condition you can do is by equality; needs expansion.

SpecifySelection transformers should probably be edited out of the graph after Slicer placement.

Right now if you have a compiled signal and an uncompiled signal that you'd at a glance think would be 
compatible for a multi-signal operation, the compiled one, which may have already had SliceSelectors place, 
might have a different schema.  Need to decide what to do about this. Maybe just accept and document it.

When making a new ordinal dim during selection, the program should validate and raise if the new loci belong to more than one earlier ordinal dim.  

In a true DAG (i.e., not a tree, with consumers that share an upstream node), `walk_graph` will create multiple SelectionSlicers.  At some point it's consumer named argument needs to be more expressive to prevent this kind of duplication.

`attach_condition_coords` only attaches coords if all loci have the condition (`conditions = reduce(and_, [set(l.conditions.keys()) for l in self.locus])`).  Should decide if that's the desired behavior.

`SliceSelection` is in an intermediate state.  After building a set of more carefully organized methods using the intersection of masks approach, I realized this was unacceptably slow for cases with regular arrays that could be handled by a single isel call, so now there is a long, insufficiently general method that handles this for the case where you're selecting child intervals that have one parent, and child and parent intervals make perfectly rectangular data. I think it is not worth trying to reorganize and make this properly general until I am ready to tackle ragged arrays, because it's only then that I will see the necessary abstraction.

In the slow path attach_condition_coords is potentially wrong when the order of child conditions varies over levels of the parent locus set -- it should be a 2D coordinate.  


## Aggregation 

Right now, if you grouped the long axis (created by AssembleArray), you can't carry over any ungrouped coords.  For example, if you had neuron and neuron type on the long axis, and then you group by neurons, neuron_type is lost Eventually you should be able to migrate that coord over to the new axis, but that will require that somewhere knowledge is encoded about how to migrate them.

You should be able to calculate a simultaneous mean (i.e. unweighted by number of members of a group.)

`reduce()` and `ReduceDim` knowingly violate the API constitution's constructor-mirroring rule. `ReduceDim` supports weighted reduction, but it hasn't been decided how to deal with weights in the fluent API.


## Plotting

The plot specification currently treats multiple x or y figure labels and overlapping labels on the same panel side as a conflict. The user should be able to specify multiple x or y axes on different positions, as well as major and minor axes on the same side.

Automatic layout does not handle data with no condition coordinates. It should
produce one panel, but `LayoutResolver` currently attempts to reduce an empty
collection of condition levels.
