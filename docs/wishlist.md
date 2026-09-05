# K-Onda Wishlist

This document records granular capabilities that may be useful but are not yet
prioritized or promised. Broad, prioritized directions belong in the
[Roadmap](plans/Roadmap.md); deficiencies in existing behavior belong in
[known issues](known_issues.md).

## Calculations

### Normalization

- Support normalization within groups defined by repeated values of an arbitrary
  user-named coordinate.

### Histograms

- Support weighted histograms after deciding how data-aligned weights enter the
  API. Prefer treating weights as a second aligned Signal, with possible
  Dataset-key convenience, rather than embedding a potentially large array in
  calculator configuration. A previously dormant internal path handled either
  one weight per value on the histogrammed axis or a full-shape weight array,
  but no public API could reach it.

## Schema infrastructure

- Let coordinate schemas declare a value-type family, such as numeric, datetime,
  categorical, or string. Define compatibility between type families centrally so
  operations can determine whether two coordinates are comparable from their
  schemas rather than discovering incompatibility while comparing materialized
  values.

## Execution diagnostics

- Make dimension-related errors explain the relevant schema history. In
  particular, when `ExtractFeatures` receives a non-scalar feature result, report
  the feature name, group or member when available, and the result's remaining
  dimension names and sizes. Add a reusable diagnostic helper that can walk
  backward through a signal graph, compare input and output schemas, and report
  where an unexpected dimension was introduced or inherited. Phrase this as
  dimension provenance rather than automatically blaming the introducing
  transformer; a downstream calculator may instead have failed to reduce the
  dimension. For example, the diagnostic could report that a remaining
  `"spikes"` dimension was introduced by `StackSignals(dim="spikes")`.

- Make exact-alignment failures report the operands that disagree. An xarray
  error such as `AlignmentError: cannot align objects with join='exact' where
  index/labels/sizes are not equal along ... 'spike'` identifies the affected
  dimension but does not show each operand's size, coordinate values or compact
  coordinate summary, signal identity, or relevant dimension provenance. K-Onda
  should add that context while preserving the original alignment exception.


## Selection

- I removed intervals and exclude initial from `Rate` because I decided they should be
  `select`'s responsibility. These parameters used to accept callables. Eventually
  it would be nice if `select` could accept a callable.

## Plotting

- Add a public figure-size specification. `Render` currently uses an 8-by-8
  default and checks for an undocumented `PlotNode.figsize` attribute, but no
  plotting method sets it.
