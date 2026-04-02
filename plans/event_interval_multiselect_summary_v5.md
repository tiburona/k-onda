# K-Onda: Events, Intervals, and Multi-Select Design Summary

*v5 — Revised March 2026. Incorporates Pynapple comparative review findings and implementation decisions from Selector refactor work.*

## Minimal Demo Target: Peri-Stimulus Time Histogram (PSTH)

The first end-to-end demo is a PSTH: take spike data, slice around stimulus events, bin into counts, group by experimental conditions, and average across presentations. This traces a critical path through the design:

```python
# The demo in code — single session
psth = (
    spike_signal
    .select(tone_epoch)                                         # restrict to recording epoch (pushdown)
    .select(pip_events, window=(-100, 300), new_dim='pip')      # slice around each stimulus (local)
    .count(bin_size=5)                                           # spikes → binned counts
    .group_by('stimulus', 'intensity')                           # group by conditions
    .mean(dim='pip')                                             # average within each condition
)

# experiment-level — using lazy epoch references and condition filtering
Experiment.from_config(some_config)
    .initialize()
    .all_neurons
    .classify_neurons(some_config)
    .select('epochs', stimulus='tone', new_dim='trial')
    .select('events', window=(-0.05, 0.3), new_dim='pip')
    .count(some_config)
    .mean([{'across':'pip'}, {'across': 'trial', 'group_by': 'stimulus'},
        {'across': 'neuron', 'group_by': 'neuron_type'},
        {'across': 'animal', 'group_by': 'treatment_group'}])
```

Note: `'pip'` is the user's name for the dimension — it could be `'stim'`, `'presentation'`, `'block'`, or anything meaningful to the experiment. This name is what makes the relative coordinate syntax work: `pip_time`, `stim_time`, `block_time`, etc. are all derived from whatever the user chose.

Each step below is marked **[PSTH-critical]** if it's on this critical path, or **[Deferred]** if it's important but not blocking the demo.

---

## Step 1: Marker and Interval Base Classes — [PSTH-critical]

### Marker

`Marker` is the dim-agnostic representation of a single point on any axis. It carries:

- `dim` — which dimension (e.g. `'time'`, `'frequency'`)
- `value` — the scalar coordinate
- `index` — optional integer identifier
- `conditions` — a dict of named condition variables (e.g. `{'stimulus': 'tone', 'frequency': '4kHz'}`)

`Marker.to_interval(window)` takes a `(pre, post)` tuple and returns an `Interval` with `lo = value + pre`, `hi = value + post`, inheriting the Marker's `conditions` and `index`.

**Condition key validation:** Condition keys are validated at construction time against reserved dim names (from `DIM_DEFAULT_UNITS` in `loci/core.py`). A condition named `'time'` or `'frequency'` raises immediately. This prevents ambiguity when `select` later partitions kwargs into dims vs conditions.

**PSTH need:** Events are Markers. The conditions dict on each event is how stimulus identity and other experimental variables get attached to each presentation.

### Event

`Event` inherits from `Marker` with `dim='time'`. Adds `session`. Aliases `value` as `timestamp` via a property.

`Event.to_interval(window)` returns an `Epoch` (not a plain `Interval`), since it has session context. This override may or may not be worth it — simpler alternative is to let it return a plain `Interval` and have Selector treat them uniformly.

Epoch requires `session` for eventual absolute-time resolution. A lightweight alternative for relative epochs: construct `Interval('time', ...)` directly or use a thin `RelativeEpoch` alias.

### Interval

`Interval` is the dim-agnostic representation of a range on any axis. It carries:

- `dim` — which dimension it selects on
- `lo`, `hi` — bounds
- `units` — optional
- `index` — optional integer identifier
- `conditions` — a dict of named condition variables, same pattern as Marker

### Epoch

`Epoch` inherits from `Interval` with `dim='time'`. Adds `session`. Computes `lo`/`hi` from `onset` and `duration`. The old `epoch_type` single string is replaced by `conditions`.

### FrequencyBand

`FrequencyBand` inherits from `Interval` with `dim='frequency'`.

Any future selection type (spatial region, etc.) can also inherit from `Interval`.

### Compound Types — [Deferred]

`Marker` and `Interval` are single-dim primitives. For multi-dimensional selection (e.g. spatial regions with x and y, or a point on a calcium imaging field), thin compound wrappers bundle single-dim primitives:

`CompoundInterval` holds a list of `Interval` objects on different dims. Selector already supports multi-dim selection via `**dim_bounds`; `CompoundInterval` just provides a convenience object that unpacks into that:

```python
class CompoundInterval:
    def __init__(self, intervals):
        self.intervals = intervals  # list of Intervals on different dims
    
    @property
    def dim_bounds(self):
        return {i.dim: (i.lo, i.hi) for i in self.intervals}
```

`CompoundMarker` holds a list of `Marker` objects on different dims. `to_interval(windows)` takes a dict of `{dim: (pre, post)}` and returns a `CompoundInterval`:

```python
class CompoundMarker:
    def __init__(self, markers):
        self.markers = markers
    
    def to_interval(self, windows):
        return CompoundInterval([
            m.to_interval(windows[m.dim]) for m in self.markers
        ])
```

These are not needed for the current ephys work but the path is clear for use cases like behavioral tracking (mouse position in cage/maze → select region → mask → detect epochs) or spatial neural data (calcium imaging fields).

**Note:** Multi-dim selection (time + frequency simultaneously) is temporarily regressed during the Selector refactor. Will be restored once multi-select is working end-to-end, likely via CompoundInterval.

---

## Step 2: IntervalSet — Collections with Set Algebra

An `IntervalSet` is a collection of `Interval` objects on the same dimension, with set operations built in. This is not a signal — it's a description of *which regions* of a dimension are selected or valid. Think of it as a stencil you compose before applying it to data.

### Minimal form — [PSTH-critical]

For the PSTH demo, the IntervalSet only needs to be a structured collection with conditions — essentially an `EventCollection` that can convert to intervals via `to_interval_set(window)`, and an `IntervalSet` that the Selector can iterate over. Condition filtering via `.where(**conditions)` is needed to support condition selection in the `select` API. No set algebra needed yet.

### Full form — [Deferred]

In practice you constantly need to combine interval collections before using them to slice signals: "theta epochs minus artifact epochs, restricted to correct trials." That requires algebra on the collections themselves — not just iteration.

**Set algebra** (returns a new IntervalSet):
- `union(other)` / `|` — combine two sets of intervals
- `intersect(other)` / `&` — keep only overlapping regions
- `difference(other)` / `-` — punch holes (e.g. remove artifact periods)

**Practical helpers** (return a new IntervalSet):
- `drop_short(min_duration)` — remove intervals shorter than a threshold
- `merge_close(gap)` — merge intervals separated by less than a gap
- `split(size)` — chop into uniform chunks

**Queries** (return other types):
- `contains(markers)` — which Markers/Events fall inside which intervals
- `total_duration` — sum of all interval lengths

### Provenance on IntervalSets

IntervalSets carry a construction log — an append-only record of operations that produced them (e.g. `[('source', 'correct_trials'), ('intersect', 'recording_epochs'), ('difference', 'artifact_epochs')]`). This is not a full lazy DAG — just metadata. When a Window node in the signal DAG consumes an IntervalSet, the construction log is folded into the signal's provenance context. The signal DAG remains the only real computation graph; the interval's history enriches the provenance record so you know not just *what* was selected but *how* the stencil was built.

### Conditions

Each interval in the set carries its own `conditions` dict. Set operations preserve conditions where it makes sense (e.g. intersection keeps conditions from both sides) and drop them where it doesn't (e.g. merging intervals with different conditions).

### Lazy by default

An IntervalSet is just a description — it doesn't touch signal data. It stays symbolic until a Selector uses it to actually slice. This fits K-Onda's lazy evaluation model: you compose your stencil, then apply it.

### Operator conventions

Arithmetic operators (`+`, `-`, `*`, `/`) are reserved for signal math. IntervalSet uses Python's set-style operators (`|`, `&`, `-`) plus named methods. No ambiguity — they live on different types.

### Subclasses

- `EpochSet` — IntervalSet with `dim='time'`, elements are Epochs
- `FrequencyBandSet` — IntervalSet with `dim='frequency'`
- `EventCollection` — a collection of Markers/Events (not intervals), with `to_interval_set(window)` to convert

---

## Step 3: Conditions Model — [PSTH-critical]

Conditions are a dict of named condition variables, not a single string or a tuple. Examples:

- Epoch conditions: `{'stimulus': 'tone', 'intensity': 'loud'}`
- Event conditions: `{'pip_freq': '4kHz'}`
- Subject conditions: `{'drug_group': 'treatment', 'sex': 'male'}`

Multiple condition variables are supported by adding keys. The semantics of within-subject vs between-subject come from *which dimension the conditions live on* (epoch/event dimension vs subject dimension), not from any special type. `group_by` in the aggregation chain references condition names directly.

**Condition keys must not collide with dim names.** This is enforced at construction time on Marker, Interval, and their subclasses. The authoritative set of reserved names comes from `DIM_DEFAULT_UNITS` in `loci/core.py` (e.g. `'time'`, `'frequency'`).

**PSTH need:** The conditions on each stimulus event (e.g. tone frequency, intensity) flow through multi-select to become xarray coordinates on the `'pip'` dimension — meaning labeled indices, not `.attrs` metadata. This is what makes them selectable and groupable: `group_by('stimulus', 'intensity')` uses xarray's native indexing to partition presentations before averaging. This is the mechanism that turns a flat pile of presentations into condition-separated PSTHs.

---

## Step 4: IntervalSignal Type Hierarchy — [PSTH-critical, under review]

When Selector multi-selects using an IntervalSet, it produces a signal with an additional named dimension corresponding to the intervals.

### Current decision: defer class proliferation

The full `EpochedSignal` / `EpochedTimeSeriesSignal` / etc. hierarchy is deferred until concrete behavioral differences justify it. For now, the signal carries a flag or attribute (e.g. `is_epoched`, or a `selection` attribute that's `None` vs an IntervalSet) and methods route internally where needed. If too many `if is_epoched` branches accumulate, that's the signal to extract subclasses — but not before.

### Planned hierarchy (if needed)

- `IntervalSignal` — parent
  - `EpochSignal` — interval dimension is time-based
    - `EpochScalarSignal` (already exists)
    - `EpochTimeSeriesSignal`
    - `EpochTimeFrequencySignal`
    - `EpochPointProcessSignal`
  - Analogous subtypes for frequency band selection if needed

### Dimension Naming

The new dimension is user-named (e.g. `'pip'`, `'stim'`, `'band'`). Each interval's `conditions` dict entries become xarray coordinates on that dimension. This is how `group_by` in the later aggregation chain finds them.

A doubly-epoched signal (e.g. blocks containing pips) is just a signal with two named interval dimensions. No special type needed — the user-supplied dim names distinguish them.

**PSTH need:** The PSTH demo produces a spike signal with an additional `'pip'` dimension, which after `count(bin_size)` becomes a time series signal with the `'pip'` dimension. Each presentation carries condition coordinates (stimulus type, intensity, etc.) that `group_by` and `mean` operate on.

### Ragged epochs

xarray requires rectangular dimensions. Variable-length epochs (common in practice) can't be directly stacked. For the PSTH demo, epochs are uniform (same window around each event). The ragged case is deferred — the plan is to detect raggedness at the point of stacking and either raise a helpful error or route to a different representation.

---

## Step 5: SelectMixin and Selector Refactor — [PSTH-critical]

### Architecture: SelectMixin as the user boundary

`SelectMixin` is the user-facing interface. `Selector` is pipeline machinery. The mixin accepts messy human input (mixed kwargs, conditions, bare coordinates, string references), interprets it, and hands Selector a fully-formed locus. Selector never sees raw kwargs.

The flow:
1. Mixin receives `**kwargs`
2. Partitions kwargs into dim bounds vs conditions (using the known dim name set from `DIM_DEFAULT_UNITS`)
3. Constructs the appropriate locus from dim bounds (if any)
4. Resolves lazy epoch references (string arguments — see Step 5b)
5. Filters by conditions via `selection.where(**conditions)` (if any)
6. Determines default mode (pushdown vs local — see below)
7. Hands Selector a clean locus object

Selector's init **rejects** raw kwargs — if it ever receives them, that's a bug in the mixin.

### SelectMixin API

```python
class SelectMixin:
    def select(self, selection=None, new_dim=None, mode=None, window=None, **kwargs):
        # kwargs are partitioned into dim_bounds and conditions
        # based on whether the key matches a known dim name
        ...
```

The `select` method unifies what was previously split across coordinate selection and condition filtering. A single call can specify both:

```python
# keyword args are partitioned: 'time' is a known dim → dim bound; 'stimulus' is not → condition
signal.select('epochs', stimulus='tone', time=[(0, 30), (50, 80)], new_dim='trial')
```

### Pushdown vs Local: Default Mode

The default mode is determined by the signal's current state — specifically, whether the signal already carries interval dimensions on the dim being selected:

- **No existing interval dimensions on this dim** → **pushdown**. This is the initial coarse restriction — "only compute within these regions." Applies whether the selection is a pre-constructed EpochSet, a string reference, or even an IntervalSet constructed on the fly.
- **Signal already has interval dimensions on this dim** → **local**. The coarse restriction has already happened; this is fine work within already-computed data (e.g. peri-event slicing within an already-restricted epoch).

This captures the typical two-layer pattern: the first `select` does the big blocks (pushdown), the second `select` does the small windows within them (local). The signal's state makes the intent unambiguous without relying on heuristics about argument form.

Override with `mode='local'` or `mode='pushdown'` when the default is wrong.

### Dual Coordinates — [PSTH-critical]

When `select` creates a named dimension from events + a window, it attaches **both** absolute and relative time coordinates to the resulting xarray data. Nothing is lost, nothing is replaced:

- **Absolute time** stays as an auxiliary coordinate — each slice retains its original timestamps (e.g. presentation 1 runs 5000–5400ms, presentation 2 runs 12300–12700ms)
- **Relative time** (the window-derived coordinate) becomes the primary axis — every slice shares a common axis (e.g. -100 to +300ms relative to stimulus onset)

This means `mean(dim='pip')` works immediately — all presentations are already aligned on their relative time axis. No separate re-zeroing or `align()` step needed. And absolute time is still there if you need it (e.g. checking whether two events overlapped in real time).

### User-Named Relative Coordinate Syntax

When `select` creates a named dimension (via `new_dim='...'`), that name is what unlocks relative coordinate access in downstream `select` calls. The pattern is `{dim_name}_{coord_dim}`, where `dim_name` is whatever the user chose — it's not a built-in keyword.

The key contrast is with plain coordinate kwargs, which are always absolute:

```python
# new_dim='pip' creates the named dimension — this is what makes 'pip_time' available
signal.select(pip_events, window=(-100, 300), new_dim='pip')

# ABSOLUTE: time=(50, 150) means absolute recording time — milliseconds 50 to 150
signal.select(pip_events, window=(-100, 300), new_dim='pip').select(time=(50, 150))

# RELATIVE: pip_time=(50, 150) means 50 to 150ms relative to each pip's onset
signal.select(pip_events, window=(-100, 300), new_dim='pip').select(pip_time=(50, 150))
```

The name is entirely user-driven. If they'd written `new_dim='stim'`, the relative accessor would be `stim_time`. If `new_dim='block'`, it'd be `block_time`:

```python
signal.select(pip_events, window=(-100, 300), new_dim='stim').select(stim_time=(50, 150))
signal.select(tone_epochs, new_dim='block').select(block_time=(5000, 10000))
```

With two levels of nesting, the names are unambiguous — the user can reference either level:

```python
signal.select(tone_epochs, new_dim='block').select(pip_events, window=(-100, 200), new_dim='pip')
# block_time=(5000, 10000)  — relative to each block's onset
# pip_time=(50, 150)        — relative to each pip's onset
```

The parsing rule: split on the last underscore, look up the first part as a known dimension name, and use the second part as the coordinate dimension. This is not time-specific — `pip_frequency` would also work if a multi-select had been done on the frequency axis.

### Examples

```python
# === PSTH critical path ===

# restrict to epoch, then slice around each stimulus event
signal.select(tone_epoch).select(pip_events, window=(-100, 300), new_dim='pip')

# with lazy reference and condition filtering
signal.select('epochs', stimulus='tone', new_dim='trial').select('events', window=(-0.05, 0.3), new_dim='pip')

# === Absolute vs relative selection ===

# absolute: select a region of absolute recording time
signal.select(tone_epoch).select(pip_events, window=(-100, 300), new_dim='pip').select(time=(5000, 5200))

# relative: select 50–150ms after each pip onset (new_dim='pip' created the dimension)
signal.select(tone_epoch).select(pip_events, window=(-100, 300), new_dim='pip').select(pip_time=(50, 150))

# === Additional forms ===

# single epoch select — current behavior, pushdown by default
signal.select(tone_epoch)

# multi epoch select — user names the dimension, pushdown by default
signal.select(tone_epochs, new_dim='block')

# nested multi-select — blocks containing pips
signal.select(tone_epochs, new_dim='block').select(pip_events, window=(-100, 200), new_dim='pip')

# relative sub-window within each block
signal.select(tone_epochs, new_dim='block').select(block_time=(50, 150))

# relative selection referencing an outer dimension through a nested select
signal.select(tone_epochs, new_dim='block').select(pip_events, window=w, new_dim='pip').select(block_time=(50, 150))

# frequency band multi-select
signal.select(frequency_bands, new_dim='band')

# composing intervals before selecting [Deferred — requires set algebra]
clean_epochs = correct_epochs & recording_epochs - artifact_epochs
signal.select(clean_epochs, new_dim='block')

# mixed kwargs: dim bounds and conditions in one call
signal.select('epochs', stimulus='tone', time=(0, 30), new_dim='trial')
```

### Tree Walk

The tree walk happens **once** regardless of how many intervals are in the collection. The `Window` node placed at each insertion point is collection-aware — it knows about all intervals. In `_apply`, it slices once per interval and stacks along the new dimension, attaching both absolute and relative coordinates and propagating conditions.

Window receives a `WindowParams` dataclass (not a raw dict) from the mixin, ensuring all required fields are present and inspectable.

### Time Handling

Both absolute and relative time are preserved as xarray coordinates. The distinction between them is made explicit through the `select` kwarg syntax: plain `time=(...)` always means absolute recording time, while `{dim_name}_time=(...)` means time relative to each element's onset on that user-created dimension.

Nested multi-selects work because absolute time stays intact — inner intervals have absolute bounds and the inner Window doesn't need to know about the outer nesting. Each named dimension independently tracks its own relative coordinate, and the `{dim_name}_time` syntax lets the user reference whichever level they need.

---

## Step 5b: Lazy Epoch/Event References — [PSTH-critical]

### The problem

Epochs and events are per-session. A fluent chain or YAML spec that describes a pipeline across the experiment can't reference a concrete EpochSet — there isn't one until the pipeline is executed against a specific session's data.

### Solution: string references resolved from context

When `select` receives a string like `'epochs'` or `'events'` instead of an actual locus object, it resolves the reference by walking up the signal's context hierarchy (signal → session → subject → experiment) and looking up the named collection on the nearest context that has it. Resolution happens at execution time, not at chain-construction time.

This is consistent with K-Onda's lazy evaluation model — the string is a symbolic reference, just like the rest of the DAG is a symbolic description. It resolves when the pipeline materializes against a specific session.

```python
# These are equivalent at execution time:
signal.select(session.epochs, stimulus='tone', new_dim='trial')
signal.select('epochs', stimulus='tone', new_dim='trial')

# The string form is what YAML uses:
# select:
#   selection: epochs
#   stimulus: tone
#   new_dim: trial
```

The string form is trivially serializable to YAML. The object form is still available for programmatic use when the user already has a concrete EpochSet in hand.

### Type distinction is the dispatch

If `selection` is a string → resolve from context. If `selection` is an Interval/IntervalSet/EventCollection → use directly. No special `EpochReference` class needed; the type of the argument is the signal.

### `for_each` still exists

The lazy reference replaces the need for `for_each` in the fluent chain, but `experiment.for_each('session', func)` remains available as a programmatic API for cases where the user wants to define a function and apply it across sessions explicitly.

---

## Step 6: Peri-Event Time via Dual Coordinates — [PSTH-critical]

### How it works

Other libraries (notably Pynapple) treat peri-event alignment as a separate operation: slice the data, then call `align()` to re-zero the time axis. In K-Onda, alignment is not a separate step — it's a natural consequence of how `select` builds the data.

When `select` creates a named dimension from events + a window, every slice gets a relative time coordinate derived from the window (e.g. -100 to +300ms). All slices share this axis. Averaging across the dimension works immediately because the relative coordinates are already aligned. Absolute timestamps are preserved as an auxiliary coordinate, so no information is lost.

### Why no `align()`

The `align()` pattern in other libraries exists because they store only absolute time and need a separate step to re-zero it. Since K-Onda attaches both coordinates at slice time, there's nothing to align — the data comes out ready for averaging, and absolute time is still there if you want it.

If users coming from Pynapple expect an `align()` method, it could exist as a no-op or a coordinate-selection convenience, but it's not part of the core design.

### The full PSTH pipeline

```python
psth = (
    spike_signal
    .select(tone_epoch)                                         # restrict to recording epoch
    .select(pip_events, window=(-100, 300), new_dim='pip')      # slice + dual coords
    .count(bin_size=5)                                           # spikes → binned counts
    .group_by('stimulus', 'intensity')                           # group by conditions
    .mean(dim='pip')                                             # average — works because
)                                                                #   all pips share relative time
```

No alignment step. The relative time coordinate exists from the moment `select` creates the `'pip'` dimension. `mean(dim='pip')` collapses across presentations, and the result is a time series on the shared -100 to +300ms axis, one per condition.

---

## Step 7: Aggregation Across Sessions and Subjects — [PSTH-critical]

### The problem

Steps 1–6 produce a PSTH for a single signal in a single session. The demo needs to go further: average across sessions within an animal, then across animals, potentially grouped by between-subject conditions (e.g. drug group).

### Where this lives

This is an experiment-level concern, not a signal-level one. A single signal doesn't know about other sessions or other animals. The `Experiment` object (or a similar orchestrator) is what knows the full structure — which sessions belong to which subjects, which subjects are in which groups.

### Possible API

```python
Experiment.from_config(some_config)
    .initialize()
    .all_neurons
    .classify_neurons(some_config)
    .select('epochs', stimulus='tone', new_dim='trial')
    .select('events', window=(-0.05, 0.3), new_dim='pip')
    .count(some_config)
    .mean([{'across':'pip'}, {'across': 'trial', 'group_by': 'stimulus'},
        {'across': 'neuron', 'group_by': 'neuron_type'},
        {'across': 'animal', 'group_by': 'treatment_group'}])

# or: .mean(hierarchical=True, group_by=['stimulus', 'neuron_type', 'treatment_group'])
```

The `mean` call takes a list of aggregation specs, each specifying which dimension to average across and optionally how to group before averaging. This chains hierarchical averaging in a single call. The alternative compact form (`hierarchical=True`) could infer the aggregation order from the dimension structure.

The `for_each` pattern is also available for programmatic use:

```python
group_psth = (
    experiment
    .for_each('session', compute_psth)       # run on each session's spikes + events
    .mean(across='session')                   # average across sessions within each subject
    .group_by('drug_group')                   # group subjects by between-subject condition
    .mean(across='subject')                   # average across subjects within each group
)
```

### What needs to be in place

- The experiment object needs to know how to iterate over sessions and pair signals with their events
- Subject-level conditions (`drug_group`, `sex`, etc.) need to live on the subject/session metadata, not on individual events
- The aggregation chain (`mean`, `group_by`) needs to work across the session and subject dimensions, not just the presentation dimension

This connects to the experiment-level orchestration design that's already being worked on. The key point for the PSTH demo: the same `group_by` / `mean` pattern that works within a signal (grouping presentations by conditions) also works across subjects (grouping subjects by conditions). The mechanism is the same — conditions as coordinates on a named dimension — just applied at a higher level.

---

## Step 8: Selection Awareness in Downstream Operations — [Deferred]

### The problem

If a signal has been restricted to certain epochs, downstream operations need to know that. A Rate calculator shouldn't count inter-trial silence. A spectrogram shouldn't treat gaps between epochs as real data.

### Why this is mostly already solved

The data got selected — so the data reflects the selection. A Rate calculator divides by the duration of the data it actually has, not by consulting a side channel. The record of what happened is the DAG itself: Window nodes are structural provenance. No redundant annotation on the signal is needed.

If an operation ever needs to inspect what selections were applied (e.g. for a methods-section export), it walks the DAG and finds the Window nodes. A lazy `selections` property on the signal that searches the DAG is a reasonable convenience, but it's a view into the graph — not a separate bookkeeping system.

### What might still be needed

The remaining question is whether there are operations where the data alone isn't sufficient — where a Calculator needs to know not just "what data do I have" but "what was the original extent of the selection that produced this data." For instance, if a Rate calculator receives data that's been sliced to an epoch but needs to know the epoch's full duration (not just the duration of the data after a sub-selection). This is an edge case worth watching for, but not worth building machinery for until it actually arises.

### Relationship to ValidityMask

K-Onda already has a ValidityMask concept for point-by-point quality flags (e.g. marking individual noisy samples). Interval-level validity (which regions are valid) is structural — it's what the data actually contains after selection. Sample-level quality within those regions is what ValidityMask handles.

---

## Suggested Implementation Order

### PSTH Critical Path

1. **Interval + Marker base classes** — refactor Epoch and FrequencyBand to inherit from Interval. Introduce Marker and Event with conditions dicts (validated against reserved dim names). EventCollection with `to_interval_set(window)`.
2. **IntervalSet with `.where()` filtering** — minimal structured collection that supports condition filtering. Construction log for provenance.
3. **Conditions model** — conditions as dicts that flow from events through multi-select to become xarray coordinates.
4. **SelectMixin + Selector refactor** — SelectMixin as user boundary: partitions kwargs into dims vs conditions, constructs loci from dim bounds, resolves string references, filters by conditions, determines default mode (pushdown vs local). Selector receives only clean locus objects. Multi-select with dual coordinates. Window receives `WindowParams` dataclass.
5. **Lazy epoch/event references** — string-based resolution from context hierarchy for YAML compatibility.
6. **Aggregation** — `group_by` + `mean` on the presentation dimension, then experiment-level `for_each` + `mean` across sessions and subjects.

At the end of this stage, the PSTH demo works end to end.

### Post-PSTH Hardening

These items make the core pipeline robust enough for real-world use. They aren't needed for the demo but will be needed almost immediately after.

7. **Baseline normalization** — z-score or percent-change relative to a pre-stimulus window (e.g. `pip_time=(-100, 0)`). A Transformer that normalizes each trial relative to its own baseline. Tight dependency on the dual coordinate machinery.
8. **Ragged epoch handling** — variable-length trials can't be stacked into rectangular xarray dimensions. Options: pad with NaN, a list-of-arrays representation, or force scalar feature extraction before stacking. Needs a concrete decision and implementation.
9. **Trial rejection after epoching** — drop individual trials from an already-epoched signal based on criteria (artifact, disengagement, spike count). This is filtering on an existing interval dimension, different from filtering the IntervalSet before slicing.
10. **IntervalSet set algebra** — union, intersection, difference, filtering helpers. Enables composing complex epoch selections before slicing.
11. **Multi-value condition selection** — support selecting across multiple values of a condition (e.g. `stimulus=['tone', 'noise']`) in `.where()` and in `select` kwargs. Eventually: negation, numeric ranges, boolean combinations, or arbitrary Python predicates.
12. **Restore multi-dim selection** — CompoundInterval to re-enable simultaneous time + frequency selection (currently regressed during refactor).
13. **Signal concatenation along existing dims** — joining two signals along a shared dimension (e.g. combining sessions). Different from multi-select (which creates a new dim).

### Analysis Toolkit

14. **Remaining calculators** — correlograms, tuning curves, value_from.
15. **NWB I/O** — additional DataSource subclasses for broader data format support.
16. **Surrogate / randomization methods** — for statistical testing.

---

## Naming Decisions

- `span` → `select` (the fluent method on SelectMixin)
- `dim` → `new_dim` (the keyword for creating a named dimension — clearer about what it does)
- `endpoints` / `dim_endpoints` → `bounds` / `dim_bounds` (avoids collision with point process terminology)
- "Reference epochs" → "baseline epochs" (frees up "ref" terminology)
- `DIM_DEFAULT_UNITS` dict in `loci/core.py` replaces tree-walking to discover dimension units

---

## Open Questions

- Should `Event.to_interval` return an `Epoch` (preserving session context) or a plain `Interval` (simpler, Selector treats them uniformly)?
- Does the single-Interval case of `select` remain a special path (no new dimension), or is it just the N=1 case of multi-select?
- How should IntervalSet set operations handle conflicting conditions? (e.g. intersecting an interval with `{'stimulus': 'tone'}` and one with `{'stimulus': 'noise'}`)
- What does `group_by` actually return? A dict-like structure mapping condition combinations to sub-signals? A single signal with an extra grouping dimension? This affects how `mean` chains after it.
- For the `{dim_name}_{coord_dim}` parsing: should it split on the last underscore (allowing dim names with underscores like `tone_pip_time`) or require dim names without underscores?

## Resolved Questions

- **IntervalSignal class hierarchy**: Deferred. Use a flag/attribute on the signal for now; extract subclasses only when concrete behavioral differences accumulate.
- **Condition filtering location**: `select` accepts condition kwargs, delegates filtering to `IntervalSet.where()`. The mixin orchestrates; the collection filters.
- **Empty condition filter result**: `select` raises a clear error if `.where()` returns an empty IntervalSet.
- **Pushdown vs local default**: If the signal already has interval dimensions on the dim being selected → local. If it doesn't → pushdown. Override with `mode=` kwarg.
- **Per-session epoch resolution**: String references (`'epochs'`, `'events'`) resolve from the signal's context hierarchy at execution time. No special reference class needed — the type of the argument dispatches.
- **Epoch session requirement**: Epoch keeps requiring `session` for absolute-time semantics. Lightweight alternative: `Interval('time', ...)` or `RelativeEpoch`.
