That graph nodes and signals are synonymous is persistently confusing. There needs to be a refactor in which Signal is renamed to Node and Signal either inherits Node or  Signal becomes a Dataclass attached to Node with truly Signal-specific attributes, like sampling rate. 

I should go through all the calculators and make sure that rather than have generic config dictionaries they have separate arguments for the sake of inspectable signatures (or alternatively a params DataClass that would be inspectable), but I should also give them a generic config dictionary or some kind of callable that sweeps up all their params for the sake of pretty printing the graph.


I'm missing dispatch over `SignalMap` for Transformer and the select mixin.

In addition to all input types that aren't supported yet, there's the class called NEVMixin only really reads from NEV files that were already translated to matfiles.


Spectrogram currently assumes it's computed over time; that should be generalized.

You can't `select_point_process` yet because there's not yet support for ragged arrays.

`DimBounds` is written to such that it could have multiple dims, but locus and the selector logic are not.  Multiple dim select should be restored.  

The only filter currently supported is sos; this needs to be expanded.


To think about: do I need new `AxisKind`s for the dim over which signals are stacked?  For the integer indexes created by the user?

Histogram currently accepts a string `range_source`, "session", but there should be at least one other (which could maybe supercede "session") -- the smallest enclosing container on the histogram dim.  (e.g. if user has created epochs and the histogram is over time, "epochs".)

Spectrogram currently tests      `if isinstance(n_cycles, np.ndarray):`. This is only working because the only way anyone has run spectram is after defining config variables in Python; it will break as soon as config comes from a string source -- need to test for iterables more generally.

I should add an 'absolute' option to generate_markers.

In loci/core.py generate_markers I suspect that the line that takes q.magnitude is going to break on the iterables (`offsets`, `positions`).

# TODO: architectural debt. 
Threading data_identity through the origin
        # chain means deepcopying any signal drags the entire entity graph along,
        # causing hash failures on partially-constructed DataIdentity objects
        # (deepcopy uses __new__, not __init__). Short-term fix: add __deepcopy__
        # to DataIdentity to return self. Long-term fix: store data_identity_id
        # (uid) on the root signal and resolve it via a registry, so signals can
        # still belong to a DataIdentity without holding a live reference.
