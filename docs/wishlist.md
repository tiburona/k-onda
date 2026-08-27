# K-Onda Wishlist

This document records granular capabilities that may be useful but are not yet
prioritized or promised. Broad, prioritized directions belong in the
[Roadmap](plans/Roadmap.md); deficiencies in existing behavior belong in
[known issues](known_issues.md).

## Calculations

### Normalization

- Support normalization within groups defined by repeated values of an arbitrary
  user-named coordinate.

## Schema infrastructure

- Let coordinate schemas declare a value-type family, such as numeric, datetime,
  categorical, or string. Define compatibility between type families centrally so
  operations can determine whether two coordinates are comparable from their
  schemas rather than discovering incompatibility while comparing materialized
  values.

## Execution diagnostics

- After the n-ary transformer work is complete, add structured context to errors
  raised during lazy graph execution. Use `Exception.add_note()` so the original
  exception type, message, and traceback remain intact. The shared machinery
  should distinguish the `apply_inner`, `apply`, and `materialize` stages, report
  the phase within `apply` or `materialize`, and deduplicate notes by stage rather
  than suppressing all later context. Notes should include compact information
  such as `format_call()`, key routing, signal identity, and input names,
  dimensions, dtypes, and units without printing full data values. Avoid adding a
  materialization note for every downstream node unless a future verbose mode
  explicitly requests the complete graph path.

## Plotting

- Add a public figure-size specification. `Render` currently uses an 8-by-8
  default and checks for an undocumented `PlotNode.figsize` attribute, but no
  plotting method sets it.
