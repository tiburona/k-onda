## IO 

In addition to all input types that aren't supported yet, the class called NEVMixin only really reads from NEV files that were already translated to matfiles.

This is more of a question than an issue, and probably not an answerable one until I see more examples, but I had a TODO that read "Should I eventually try to unify the helpers/properties of PhyOutput and Generic Spike Source?"  


## Model and Config

Experiment inherits from AnnotatorMixin, but is not currently setting an annotation upon adding a subject.


## Coordinates and schemas

K-Onda does not yet have a general, unit-aware policy for coordinate tolerance.
Operations currently vary between exact coordinate equality and tolerances expressed
as a number of decimal places. Decimal places do not describe a stable physical
tolerance because their meaning changes when the same coordinate is represented in
seconds, milliseconds, or another compatible unit. Coordinate schemas and shared
comparison utilities should eventually express tolerances with units or derive them
from a declared grid resolution, convert compatible units before comparison, and
distinguish trivial floating-point differences from genuinely different grids that
require alignment or resampling.

K-Onda also lacks a general policy for propagating coordinates through operations
that align multiple inputs. Arithmetic currently returns the primary input's schema,
but xarray drops non-index coordinates whose values conflict between operands. For
example, subtracting tone and pretone data preserves their shared relative-time
coordinates but drops absolute time and condition coordinates, while the arithmetic
output schema still declares those coordinates. Multi-input operations need to track
which coordinates are required to align, which are preserved because they agree,
which are transformed, and which are intentionally dropped, and then apply the same
decisions to the output schema. Arithmetic is the known concrete case, but all
transformers that rely on xarray's implicit coordinate propagation should be audited
for the same data/schema divergence.


## Graph and Signals

That graph nodes and signals are synonymous is persistently confusing. There needs to be a refactor in which Signal is renamed to Node and Signal either inherits Node or  Signal becomes a Dataclass attached to Node with truly Signal-specific attributes, like sampling rate. 

To think about: do I need new `AxisKind`s for the dim over which signals are stacked?  

A `SignalStack` can't currently be compiled.  You probably mostly wouldn't want to, but there's no reason it shouldn't have the capability.

I had `PointProcessSignal` stop inheriting from DatasetSignal because a `PointProcessSignal` doesn't *have* to be a DatasetSignal`. However some PPS's are DatasetSignals and I should come up with an inheritance model that expresses it.  

Right now `payload` is guaranteed to return a signal of the same type for `DatasetSignal`s and I don't think that's right.

`Intersection` and `ApplyMask`'s `__call__`s need to be evaluated for how they're working with keys, how they apply to stacks, and in general to be brought up-to-date with the code base.  

`Intersection` does not yet support signals sampled on genuinely different grids
(for example, a 30 fps video-derived mask and power calculated every 0.01 seconds).
It needs an explicit alignment/resampling policy and a decision about the output
grid. Coordinate tolerance should handle only trivial floating-point differences
between grids that are otherwise equivalent. It should eventually be possible to
specify that tolerance with units rather than only as a number of decimal places.
`Intersection` should also support locations described by multiple paired
coordinates (for example, repeated observations with `x` and `y` coordinates),
and define pairing semantics for repeated coordinates that cannot be matched
unambiguously by position.


## Loci

`Locus`/`LocusSet` currently has no provenance record.  This is okay for config-derived epochs/events, but insufficient for future loci derived from signals.

There are some forbidden names for loci conditions (e.g., 'time', 'frequency'). Right now the code raises ValueError but when I actually have docs the messages should be replaced with a nice link to the docs.


## Transformers

I should go through all the calculators and make sure that rather than have generic config dictionaries they have separate arguments for the sake of inspectable signatures (or alternatively a params DataClass that would be inspectable), but I should also give them a generic config dictionary or some kind of callable that sweeps up all their params into one object for the sake of pretty printing the graph.

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

`SliceSelection` constructs a relative coordinate by subtracting each selection's
absolute starting coordinate. Mathematically identical relative grids can therefore
differ by floating-point noise and fail later exact alignment. Rounding the relative
coordinate to a fixed number of decimal places is only a provisional workaround,
because the resulting tolerance depends on the coordinate's current unit. Selection
should instead produce a canonical relative grid by using the broader unit-aware
coordinate-tolerance or grid-resolution policy.

`DimBounds` is written such that it could have multiple dims, but loci and the selector logic are not.  Multiple dim select should be restored. 

You can't `select_point_process` yet because there's not yet support for ragged arrays.

The only kind of filtering by condition you can do is by equality; needs expansion.

SpecifySelection transformers should probably be edited out of the graph after Slicer placement.

Compiled and uncompiled selection pipelines currently expose schemas from different
planning stages and therefore cannot reliably be composed as operands. An uncompiled
`SpecifySelection` still reports its input schema, while compilation replaces it with
`SliceSelection`, whose schema includes the planned ordinal and relative-coordinate
dimensions. Two otherwise equivalent pipelines can consequently appear structurally
incompatible when one has been compiled and the other has not. Until selection's
planned output schema is available consistently before compilation, callers must keep
both operands symbolic and compile the combined expression, or compile both operands
before combining them.

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
