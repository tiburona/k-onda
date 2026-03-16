# K-Onda: Events, Intervals, and Multi-Select Design Summary

*Revised March 2026 to incorporate findings from Pynapple comparative review*

## Minimal Demo Target: Peri-Stimulus Time Histogram (PSTH)

The first end-to-end demo is a PSTH: take spike data from categorized neurons, select tone epochs (creating an epoch dimension), slice around stimulus events within those epochs (creating a pip dimension), bin spikes into counts, group by stimulus condition, and then average — first across pips, then across epochs, then across neurons grouped by cell type, then across subjects grouped by experimental condition.

This traces a critical path through the design. The exact pipeline API is not yet settled — the sequence of operations and the dimensions involved are what matter here, not the specific method names or syntax.

Note on dimension naming: whenever a `select` call creates a new dimension (via `dim='...'`), the name is chosen by the user — `'pip'`, `'epoch'`, `'block'`, `'stim'`, or anything meaningful to the experiment. This user-chosen name is what makes the relative coordinate syntax work (e.g. `pip_time`, `epoch_time`). See Step 5 for details.

Note on units: windows and coordinates should support pint-backed units. A window of `(-0.1, 0.3)` in seconds and `(-100, 300)` in milliseconds describe the same selection — the unit system that K-Onda already uses should carry through here.

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

**PSTH need:** Events are Markers. The conditions dict on each event is how stimulus identity and other experimental variables get attached to each presentation.

### Event

`Event` inherits from `Marker` with `dim='time'`. Adds `session`. Aliases `value` as `timestamp` via a property.

`Event.to_interval(window)` returns an `Epoch` (not a plain `Interval`), since it has session context. This override may or may not be worth it — simpler alternative is to let it return a plain `Interval` and have Selector treat them uniformly.

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

`CompoundInterval` holds a list of `Interval` objects on different dims. Selector already supports multi-dim selection via `**dim_endpoints`; `CompoundInterval` just provides a convenience object that unpacks into that:

```python
class CompoundInterval:
    def __init__(self, intervals):
        self.intervals = intervals  # list of Intervals on different dims
    
    @property
    def dim_endpoints(self):
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

---

## Step 2: IntervalSet — Collections with Set Algebra

An `IntervalSet` is a collection of `Interval` objects on the same dimension, with set operations built in. This is not a signal — it's a description of *which regions* of a dimension are selected or valid. Think of it as a stencil you compose before applying it to data.

### Minimal form — [PSTH-critical]

For the PSTH demo, the IntervalSet only needs to be a structured collection with conditions — essentially an `EventCollection` that can convert to intervals via `to_interval_set(window)`, and an `IntervalSet` that the Selector can iterate over. No set algebra needed yet.

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

**PSTH need:** The conditions on each stimulus event (e.g. tone frequency) flow through multi-select to become xarray coordinates on the pip dimension. `group_by('stimulus')` then uses those coordinates to partition presentations before averaging. This is the mechanism that turns a flat pile of presentations into condition-separated PSTHs.

---

## Step 4: IntervalSignal Type Hierarchy — [PSTH-critical]

When Selector multi-selects using an IntervalSet, it produces an `IntervalSignal` — a signal with an additional named dimension corresponding to the intervals.

### Hierarchy

- `IntervalSignal` — parent
  - `EpochSignal` — interval dimension is time-based
    - `EpochScalarSignal` (already exists)
    - `EpochTimeSeriesSignal`
    - `EpochTimeFrequencySignal`
    - `EpochPointProcessSignal`
  - Analogous subtypes for frequency band selection if needed

### Dimension Naming

The new dimension is user-named (e.g. `'pip'`, `'stim'`, `'band'`). Each interval's `conditions` dict entries become xarray coordinates on that dimension. This is how `group_by` in the later aggregation chain finds them.

A doubly-epoched signal (e.g. blocks containing pips) is just an IntervalSignal with two named interval dimensions. No special type needed — the user-supplied dim names distinguish them.

**PSTH need:** The PSTH demo starts from categorized neurons and produces an `EpochPointProcessSignal` (spike times with additional epoch and pip dimensions), which after binning becomes an `EpochTimeSeriesSignal`. Each presentation carries condition coordinates (stimulus type, etc.) that `group_by` and `mean` operate on. Neuron-level metadata (cell type) needs to propagate through these operations so it's available for grouping at the aggregation stage.

---

## Step 5: Selector Refactor — [PSTH-critical]

*Note: The `select` / `window` terminology throughout this document is provisional. The right verb and the right noun for these operations are still being worked out — the design of the mechanism is ahead of the naming.*

### API Surface

`select` (the fluent method) accepts several forms:

- **Single Interval** → current behavior, no new dimension
- **IntervalSet + `dim='name'`** → multi-select, produces IntervalSignal with named dimension
- **Events + `window=(pre, post)` + `dim='name'`** → calls `to_interval_set(window)` internally, then multi-selects
- **Relative coordinates via `{dim_name}_{coord_dim}` kwargs** → offsets endpoints relative to the named dimension's interval onsets (see Dual Coordinates below)

**PSTH need:** The third form — events + window + dim name — is the one the PSTH demo uses. It slices a window around each stimulus event and stacks the results along a user-named dimension.

### Dual Coordinates — [PSTH-critical]

When `select` creates a named dimension from events + a window, it attaches **both** absolute and relative time coordinates to the resulting xarray data. Nothing is lost, nothing is replaced:

- **Absolute time** stays as an auxiliary coordinate — each slice retains its original timestamps (e.g. presentation 1 runs 5.0–5.4s, presentation 2 runs 12.3–12.7s)
- **Relative time** (the window-derived coordinate) becomes the primary axis — every slice shares a common axis (e.g. -0.1 to +0.3s relative to stimulus onset)

This means `mean(dim='pip')` works immediately — all presentations are already aligned on their relative time axis. No separate re-zeroing or `align()` step needed. And absolute time is still there if you need it (e.g. checking whether two events overlapped in real time).

### User-Named Relative Coordinate Syntax

When `select` creates a named dimension (via `dim='...'`), that name is what unlocks relative coordinate access in downstream `select` calls. The pattern is `{dim_name}_{coord_dim}`, where `dim_name` is whatever the user chose — it's not a built-in keyword.

The key contrast is with plain coordinate kwargs, which are always absolute:

```python
# dim='pip' creates the named dimension — this is what makes 'pip_time' available
signal.select(pip_events, window=(-0.1, 0.3), dim='pip')

# ABSOLUTE: time=(50, 150) means absolute recording time — seconds 50 to 150
signal.select(pip_events, window=(-0.1, 0.3), dim='pip').select(time=(50, 150))

# RELATIVE: pip_time=(0.05, 0.15) means 0.05 to 0.15s relative to each pip's onset
signal.select(pip_events, window=(-0.1, 0.3), dim='pip').select(pip_time=(0.05, 0.15))
```

The name is entirely user-driven. If they'd written `dim='stim'`, the relative accessor would be `stim_time`. If `dim='block'`, it'd be `block_time`:

```python
signal.select(pip_events, window=(-0.1, 0.3), dim='stim').select(stim_time=(0.05, 0.15))
signal.select(tone_epochs, dim='block').select(block_time=(5, 10))
```

With two levels of nesting, the names are unambiguous — the user can reference either level:

```python
signal.select(tone_epochs, dim='block').select(pip_events, window=(-0.1, 0.2), dim='pip')
# block_time=(5, 10)      — relative to each block's onset
# pip_time=(0.05, 0.15)   — relative to each pip's onset
```

The parsing rule: split on the last underscore, look up the first part as a known dimension name, and use the second part as the coordinate dimension. This is not time-specific — `pip_frequency` would also work if a multi-select had been done on the frequency axis.

### Examples

```python
# === PSTH critical path ===

# select tone epochs (creating an epoch dimension), then slice around each stimulus event
signal.select(tone_epochs, dim='epoch').select(pip_events, window=(-0.1, 0.3), dim='pip')

# === Absolute vs relative selection ===

# absolute: select a region of absolute recording time
signal.select(tone_epochs, dim='epoch').select(pip_events, window=(-0.1, 0.3), dim='pip').select(time=(5.0, 5.2))

# relative: select 0.05–0.15s after each pip onset (dim='pip' created the dimension)
signal.select(tone_epochs, dim='epoch').select(pip_events, window=(-0.1, 0.3), dim='pip').select(pip_time=(0.05, 0.15))

# === Additional forms ===

# single epoch select — current behavior
signal.select(tone_epoch)

# multi epoch select — user names the dimension
signal.select(tone_epochs, dim='block')

# nested multi-select — blocks containing pips
signal.select(tone_epochs, dim='block').select(pip_events, window=(-0.1, 0.2), dim='pip')

# relative sub-window within each block
signal.select(tone_epochs, dim='block').select(block_time=(0.05, 0.15))

# relative selection referencing an outer dimension through a nested select
signal.select(tone_epochs, dim='block').select(pip_events, window=w, dim='pip').select(block_time=(0.05, 0.15))

# frequency band multi-select
signal.select(frequency_bands, dim='band')

# composing intervals before selecting [Deferred — requires set algebra]
clean_epochs = correct_epochs & recording_epochs - artifact_epochs
signal.select(clean_epochs, dim='block')
```

### Tree Walk

The tree walk happens **once** regardless of how many intervals are in the collection. The `Window` node placed at each insertion point is collection-aware — it knows about all intervals. In `_apply`, it slices once per interval and stacks along the new dimension, attaching both absolute and relative coordinates and propagating conditions.

### Time Handling

Both absolute and relative time are preserved as xarray coordinates. The distinction between them is made explicit through the `select` kwarg syntax: plain `time=(...)` always means absolute recording time, while `{dim_name}_time=(...)` means time relative to each element's onset on that user-created dimension.

Nested multi-selects work because absolute time stays intact — inner intervals have absolute bounds and the inner Window doesn't need to know about the outer nesting. Each named dimension independently tracks its own relative coordinate, and the `{dim_name}_time` syntax lets the user reference whichever level they need.

---

## Step 6: Peri-Event Time via Dual Coordinates — [PSTH-critical]

### How it works

Other libraries (notably Pynapple) treat peri-event alignment as a separate operation: slice the data, then call `align()` to re-zero the time axis. In K-Onda, alignment is not a separate step — it's a natural consequence of how `select` builds the data.

When `select` creates a named dimension from events + a window, every slice gets a relative time coordinate derived from the window (e.g. -0.1 to +0.3s). All slices share this axis. Averaging across the dimension works immediately because the relative coordinates are already aligned. Absolute timestamps are preserved as an auxiliary coordinate, so no information is lost.

### Why no `align()`

The `align()` pattern in other libraries exists because they store only absolute time and need a separate step to re-zero it. Since K-Onda attaches both coordinates at slice time, there's nothing to align — the data comes out ready for averaging, and absolute time is still there if you want it.

If users coming from Pynapple expect an `align()` method, it could exist as a no-op or a coordinate-selection convenience, but it's not part of the core design.

### The full PSTH pipeline

The sequence of operations for the PSTH demo is: select tone epochs (multi-select, creating an epoch dimension) → select around pip events within those epochs (multi-select, creating a pip dimension) → bin spikes into counts → group by stimulus condition → average across pips → average across epochs. The relative time coordinate exists from the moment the pip dimension is created, so averaging across pips produces a time series on a shared axis (e.g. -0.1 to +0.3s relative to stimulus onset) without a separate alignment step.

---

## Step 7: Aggregation Across Sessions and Subjects — [PSTH-critical]

### The problem

Steps 1–6 produce a PSTH for a single neuron in a single session. The demo needs to go further: average across neurons grouped by cell type, then across sessions, then across subjects grouped by experimental condition (e.g. drug group).

### Where this lives

This is an experiment-level concern, not a signal-level one. A single signal doesn't know about other neurons, other sessions, or other animals. The `Experiment` object (or a similar orchestrator) is what knows the full structure — which neurons are which type, which sessions belong to which subjects, which subjects are in which groups.

### Averaging order

The PSTH averaging needs to happen in a specific order, collapsing dimensions from innermost to outermost:

1. **Across pips** (within each epoch) — average stimulus presentations within a single epoch, grouped by stimulus condition
2. **Across epochs** (within each neuron) — average across recording epochs, grouped by epoch type
3. **Across neurons** (within each subject/session) — average across neurons, grouped by cell type. This requires neuron_type to be propagated as metadata on the signal — consistent with the Pynapple review's recommendation that metadata should travel with the data through operations.
4. **Across subjects** — average across animals, grouped by between-subject conditions like drug group

### Design questions

The current codebase has separate ReduceDim and Aggregate calculators for within-signal and across-signal averaging. From the user's perspective, the distinction between "mean across a dimension of my data" and "mean across multiple signals from different sessions" is an implementation detail they shouldn't need to track. A single `mean` verb that routes to the right mechanism based on context would be cleaner.

The `group_by` / `mean` pattern should work the same way at every level: group by conditions that live on a dimension, then collapse that dimension. Whether the dimension is pips within a signal or subjects across an experiment, the mechanism is the same — conditions as coordinates on a named dimension. The API for expressing this is not yet settled.

### What needs to be in place

- The experiment object needs to know how to iterate over sessions and pair signals with their events
- Neuron-level metadata (cell type) needs to propagate through signal operations so it's available for grouping
- Subject-level conditions (drug_group, sex, etc.) need to live on the subject/session metadata, not on individual events
- The aggregation chain needs to work uniformly across all these dimensions

---

## Step 8: Interval-Based Validity Propagation — [Deferred]

*Pynapple calls this concept `time_support`. K-Onda will find a better name — `validity_intervals`, `valid_regions`, or something else. The name `time_support` is used below only when referring to Pynapple's implementation.*

### The problem

If a signal has been restricted to certain epochs, downstream operations need to know that. A Rate calculator shouldn't count inter-trial silence. A spectrogram shouldn't treat gaps between epochs as real data. Currently, this information is implicit — the user has to remember which signals have been restricted and handle it manually.

### The idea

Every signal can optionally carry an IntervalSet describing where the signal's data is valid. When a `restrict` or `select` operation narrows a signal to certain epochs, the resulting signal's validity intervals update automatically. Downstream Calculators and Transformers can check them and operate only on valid regions without the user having to think about it.

### Relationship to ValidityMask

K-Onda already has a ValidityMask concept for point-by-point quality flags (e.g. marking individual noisy samples). Interval-based validity is complementary — it describes validity at the *interval* level (entire regions of valid/invalid time), while ValidityMask handles sample-level quality within those valid regions. A signal might have validity intervals saying "epochs 1–47 are valid" and a ValidityMask saying "but sample 3042 in epoch 12 is an artifact."

The two reinforce each other: validity intervals define the coarse structure (which regions to analyze), ValidityMask handles the fine grain within those regions.

### How it propagates

- `select(interval_set, ...)` → output validity intervals are the intersection of the input's validity intervals with the selected intervals
- Calculators that produce scalar results (like Rate) use validity intervals to determine what time counts as "real" when computing rates or averages
- The Schema could carry validity interval information so it's visible before materialization

### What this enables

Once validity intervals propagate, a common pattern like "restrict to correct epochs, then compute firing rate" becomes safe by default — the Rate calculator sees only the valid time and divides by the right denominator without the user having to pass epoch information separately.

---

## Suggested Implementation Order

### Phase 1: PSTH Critical Path

1. **Interval + Marker base classes** — refactor Epoch and FrequencyBand to inherit from Interval. Introduce Marker and Event with conditions dicts. EventCollection with `to_interval_set(window)`.
2. **Conditions model** — conditions as dicts that flow from events through multi-select to become xarray coordinates.
3. **IntervalSignal type hierarchy** — define EpochPointProcessSignal and EpochTimeSeriesSignal, and what coordinates they carry. This is where conditions become groupable.
4. **Selector refactor (multi-select with dual coordinates)** — `select` accepts EventCollection + window + dim name, produces IntervalSignal with both absolute and relative time coordinates. User-named `{dim_name}_time` syntax for downstream relative selection.
5. **Aggregation** — `group_by` + `mean` that works uniformly across pips, epochs, neurons (by type), and subjects (by group). A single `mean` verb that routes to the right internal mechanism (ReduceDim vs Aggregate) based on context.

At the end of Phase 1, the PSTH demo works end to end.

### Phase 2: Robustness and Expressiveness

6. **IntervalSet set algebra** — union, intersection, difference, filtering helpers. Enables composing complex epoch selections before slicing.
7. **Interval-based validity propagation** — attach IntervalSets to signals as validity descriptors, propagate through operations.

### Phase 3: Analysis Toolkit

8. **Remaining calculators** — correlograms, tuning curves, value_from.
9. **NWB I/O** — additional DataSource subclasses for broader data format support.
10. **Surrogate / randomization methods** — for statistical testing.

---

## Open Questions

- Should `Event.to_interval` return an `Epoch` (preserving session context) or a plain `Interval` (simpler, Selector treats them uniformly)?
- Does the single-Interval case of `select` remain a special path (no new dimension), or is it just the N=1 case of multi-select?
- How should IntervalSet set operations handle conflicting conditions? (e.g. intersecting an interval with `{'stimulus': 'tone'}` and one with `{'stimulus': 'noise'}`)
- Should interval-based validity be optional on all signals, or mandatory on temporal signals?
- What does `group_by` actually return? A dict-like structure mapping condition combinations to sub-signals? A single signal with an extra grouping dimension? This affects how `mean` chains after it.
- Should `mean` be a single smart verb that routes to ReduceDim (within a signal) or Aggregate (across signals) based on context, so the user doesn't need to know which internal mechanism is being used?
- For the `{dim_name}_{coord_dim}` parsing: should it split on the last underscore (allowing dim names with underscores like `tone_pip_time`) or require dim names without underscores?
