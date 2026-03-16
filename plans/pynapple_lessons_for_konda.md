# What K-Onda Should Learn from Pynapple

*Synthesized from independent reviews by Codex and Claude Code, March 2026*

## The Big Picture

Both reviews arrived at the same core conclusion: K-Onda's architecture (lazy evaluation, symbolic DAG, xarray backbone, provenance tracking) is more powerful than Pynapple's. Don't trade any of that away. But Pynapple has solved a set of *domain-level* problems — common neuroscience operations — that K-Onda hasn't touched yet. The goal is to absorb those domain solutions into K-Onda's richer framework.


## High Priority: Architectural Gaps

### 1. Composable IntervalSet

**Both reviews flagged this as the single most important thing to build.**

Right now, K-Onda can slice a signal to an epoch. But in practice you constantly need to *compose* epochs before slicing: "theta epochs minus artifact epochs, restricted to correct trials." That requires set operations on interval collections — intersection, union, difference — plus practical helpers like "drop intervals shorter than 200ms" and "merge intervals separated by less than 50ms."

This fits naturally into K-Onda's design. An IntervalSet can stay symbolic/lazy — it's just a description of which time regions to use, and it doesn't need to touch signal data until a Selector actually slices. The existing Interval base class and the planned IntervalCollection are the right starting points; the gap is the algebra on top.

**What it unlocks:** peri-event analysis, trial-based analyses, artifact rejection, condition-selective computation — basically everything downstream depends on being able to compose intervals cleanly.

### 2. Validity That Propagates Automatically

In Pynapple, every time series carries an IntervalSet called `time_support` that says "here's where this signal is actually valid." (K-Onda will use a different name for this concept.) When you restrict a signal to certain epochs, the resulting signal's validity intervals update automatically. Downstream operations respect them without the user having to think about it.

K-Onda has the ValidityMask concept, but it's not yet interval-based and doesn't propagate through the pipeline automatically. The goal: if a signal has been restricted to correct trials, a downstream Rate calculator shouldn't accidentally count inter-trial silence. The Schema could carry interval information, and restrict operations could update it.

This is tightly coupled to the IntervalSet work — once you have composable intervals, attaching them to signals as "valid regions" is a natural next step.

### 3. Peri-Event Alignment

Aligning neural activity to behavioral events (stimulus onset, reward delivery, lick times) is one of the most common operations in systems neuroscience. You take a signal, a set of reference timestamps, and a window, and you get back a signal re-indexed relative to each event.

K-Onda has nothing here yet. This maps cleanly to a Calculator that takes a signal + an EventCollection + a window and produces an IntervalSignal (which is already in the design doc). The multi-select `span` syntax already covers the slicing part; peri-event alignment adds the re-zeroing (expressing time relative to each event's onset rather than absolute time).

### 4. Spike-Specific Operations

Two operations came up repeatedly:

- **Binned spike counts** (`count(bin_size)`): Convert spike timestamps into a binned rate time series. This is probably the single most-used method in Pynapple — binned spike trains are the input to most population analyses. **Note:** Both reviews flagged this as missing, but K-Onda's existing Histogram calculator already covers the underlying computation. The real gap is a friendly `count(bin_size)` method on PointProcessSignal that calls the Histogram calculator under the hood — this is a verb-layer issue (#5 below), not a missing-functionality issue.

- **Correlograms**: Auto- and cross-correlograms between spike trains, essential for characterizing synchrony, refractory periods, and functional connectivity. This is a clean Calculator that takes one or two PointProcessSignals and returns a distribution or time series.


## Medium Priority: Important But Not Blocking

### 5. Neuroscience-Friendly Verbs

Both reviews emphasized that users should be able to say things like `restrict`, `count`, `align_to`, `bin` without needing to understand the generic pipeline machinery. The internal architecture can stay generic (Calculators, Transformers, Selectors), but the public API should offer domain-specific method names that feel natural to a neuroscientist. Think of it as a thin vocabulary layer on top of the powerful engine.

This connects to the fluent method chaining that's already implemented — extending it with neuroscience-specific method names on the appropriate signal types.

### 6. NWB I/O

NWB (Neurodata Without Borders) is the emerging standard format for sharing neuroscience data. Many labs use it, and most public data repositories require it. K-Onda currently reads BlackRock .nsx and Phy output. Adding NWB DataSource subclasses would dramatically expand what data K-Onda can work with.

### 7. Tuning Curves

Computing firing rate as a function of a behavioral variable (position, head direction, stimulus identity) is fundamental to systems neuroscience. This maps to a Calculator that takes a PointProcessSignal and a behavioral TimeSeriesSignal and returns something like an IndexedSignal.

### 8. Surrogate / Randomization Methods

Statistical testing requires surrogate spike trains (jittered timestamps, circular shuffles, Poisson resampling). This maps to a Transformer that takes a PointProcessSignal and returns a permuted version — you'd run it N times to build a null distribution. Not urgent, but important before K-Onda can support real statistical analyses.

### 9. Temporal Alignment Between Signals (value_from)

Aligning one time series to another's timestamps — for example, extracting a behavioral variable's value at each spike time. A Calculator that takes two signals and samples one at the other's timestamps. Occasionally needed but not a priority.


## Medium Priority: Design Patterns

### 10. Metadata Propagation Through Operations

Pynapple automatically carries metadata (like neuron labels, brain region tags) through operations like `restrict()`. K-Onda has the DataIdentity/Annotation system for tracking what things are, and the conditions dict for experimental variables. The question is whether signal operations should automatically carry annotations forward. Worth designing but not blocking.

### 11. Small Public Core

Pynapple's "five core objects" discipline is good UX thinking. Internally K-Onda can have a rich type hierarchy, but the public-facing API should feel small: a handful of signal types, a handful of interval types, and clear verbs for the common operations. Users shouldn't need to know about the DAG, the tree walk, or the Calculator framework to do standard analyses.


## What NOT to Copy

- **Eager computation.** K-Onda's lazy DAG is more powerful. Don't trade it for Pynapple's simplicity.
- **Pandas inheritance.** Pynapple's Tsd extending pandas Series causes API leakage and inheritance headaches. K-Onda's xarray approach is cleaner.
- **Frozen/minimal core philosophy.** Pynapple's stability-first constraint makes sense for a community library that's already shipped. K-Onda is still being designed — don't prematurely freeze it.
- **Point-only validity masks.** Don't model data quality as only pointwise masks. Combine the existing ValidityMask work with interval-based validity — that's where the two ideas reinforce each other.


## Suggested Build Order

Both reviews converge on roughly the same sequence:

1. **IntervalSet with set algebra** — the foundation everything else depends on
2. **Interval-based validity propagation** — makes restrict and slice operations safe by default
3. **Binned spike counts + peri-event alignment** — the most-used operations in practice
4. **Metadata-aware grouping and aggregation** — connects conditions on intervals to the group_by/aggregation chain
5. **Remaining calculators** (correlograms, tuning curves, value_from) — fill out the analysis toolkit
6. **NWB I/O + surrogate methods** — portability and statistical testing

Steps 1–2 align directly with the existing implementation plan in the Events/Intervals design doc (Interval base classes → IntervalSignal types → Selector refactor). Steps 3–4 extend it into the spike processing and aggregation work that's already on the horizon.


## Where K-Onda Is Already Ahead

Both reviews noted that K-Onda's existing design already converges with several Pynapple ideas:

- **Conditions as dicts on intervals/events** — this is exactly Pynapple's metadata-on-data pattern
- **Multi-select producing named dimensions** — this is more expressive than Pynapple's approach
- **Fluent chaining** — Pynapple doesn't have this; K-Onda's `signal.scale().filter().normalize()` syntax is a genuine UX advantage
- **Symbolic execution + provenance** — K-Onda's differentiator; Pynapple doesn't track computation history at all
- **xarray backbone** — gives labeled dimensions, units support, and dask compatibility for free
