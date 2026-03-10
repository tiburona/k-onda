# CLAUDE.md — guidance for AI assistants working on K-Onda

## Read this first

Read the README before suggesting edits to the codebase. The ten numbered principles there are the design constitution of the project. When a suggestion would violate one of them, say so — don't quietly work around it.

## Raising concerns

If a request seems architecturally wrong, conflicts with the README principles, or has significant tradeoffs, say why before proposing an implementation plan. A one-sentence flag upfront is much less disruptive than mid-implementation course corrections.

## Role of AI in this project

The developer writes her own code except for rote boilerplate. Favor explanation, review, and feedback over generating large amounts of novel code. When code generation is appropriate, keep it small and targeted.

## Architecture overview

K-Onda implements a lazy, symbolic data pipeline for electrophysiology analysis. The core idea: pipelines are constructed as graphs of nodes; data is computed only when `.data` is accessed.

### Key abstractions

**`Signal`** ([src/k_onda/signals/core.py](src/k_onda/signals/core.py))
The fundamental pipeline node. Holds `inputs` (upstream signals), a `transform` callable, and a `data_schema`. Data is computed lazily on `.data` access and optionally cached. Subclasses: `TimeSeriesSignal`, `TimeFrequencySignal`, `PointProcessSignal`, `BinarySignal`, `SignalStack`.

**`Transformer` / `Calculator`** ([src/k_onda/transformers/core.py](src/k_onda/transformers/core.py))
Callable objects that consume a Signal and return a new Signal. Can operate on a single Signal, a `Collection`, or a `GroupedCollection` — the dispatch is handled internally. `Calculator` adds data validation and a `key_spec` mechanism for operating on named variables inside `xr.Dataset`-backed signals.

**`Schema` / `DatasetSchema`** ([src/k_onda/central.py](src/k_onda/central.py))
Lightweight dimension metadata that travels with signals through the pipeline without materializing data.

**`Collection` / `GroupedCollection`** ([src/k_onda/sources/](src/k_onda/sources/))
Containers for multiple signals. Transformers broadcast over them automatically.

**`SignalStack`** ([src/k_onda/signals/core.py](src/k_onda/signals/core.py))
Vectorized form of a Collection — signals stacked along a new dimension for performance.

**Mixin system** ([src/k_onda/transformers/transformer_mixins.py](src/k_onda/transformers/transformer_mixins.py))
`CalculateMixin`, `SelectMixin`, `UnstackMixin`, `IntersectionMixin` — compose the fluent API onto Signal and SignalStack.

### Data representation
- `xr.DataArray` and `xr.Dataset` throughout, with `pint` units via `pint_xarray`
- Unit registry and domain constants in [src/k_onda/central.py](src/k_onda/central.py)
- Provenance tracked as a generation graph via [src/k_onda/graph/traversal.py](src/k_onda/graph/traversal.py)

### Source modules
- [src/k_onda/sources/lfp_sources.py](src/k_onda/sources/lfp_sources.py) — LFP signal sources
- [src/k_onda/sources/spike_sources.py](src/k_onda/sources/spike_sources.py) — spike/neuron sources
- [src/k_onda/sources/core.py](src/k_onda/sources/core.py) — Collection, GroupedCollection

### Transformer modules
- [src/k_onda/transformers/](src/k_onda/transformers/) — spectral, filter, magnitude, feature, event, mask, selector, data_shape transformers
