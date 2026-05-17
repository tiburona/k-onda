## Architecture Overview

K-Onda implements a lazy, symbolic data pipeline for electrophysiology analysis.
Pipelines are constructed as graphs of nodes; data is computed only when `.data`
is accessed.

### Core Concepts

#### Experimental Data Model

**`Experiment` / `Subject` / `Session`** ([experiment.py](../src/k_onda/model/experiment.py), [subject.py](../src/k_onda/model/subject.py), [session.py](../src/k_onda/model/session.py))
The mutable experimental context. An `Experiment` is configured from YAML or JSON, creates subjects and sessions, owns unit definitions, and provides access
to configured epochs, events, data sources, and identities.

**`DataSource` / `DataComponent` / `DataIdentity`** ([sources/core.py](../src/k_onda/sources/core.py))
`DataSource` represents a configured file or resource belonging to a session.
`DataComponent` is the concrete loadable unit inside that source, such as one LFP channel/region or one spike cluster; it can be converted into a source `Signal`, `DataIdentity` groups one or more components that represent the same experimental entity, such as a neuron or brain region, so analysis can operate on biological identities rather than only on storage layout.

#### Pipeline Graph

**`Signal`** ([signals/core.py](../src/k_onda/signals/core.py))
The fundamental pipeline node. Holds `inputs` (upstream signals), a `transform` callable, and a `data_schema`. Data is computed lazily on `.data` access and optionally cached. Signal subclasses distinguish major data shapes, such as time series, time-frequency data, point processes, datasets, binary masks, and indexed feature outputs.

**`Transformer` / `Calculator`** ([transformers/core.py](../src/k_onda/transformers/core.py))
Callable objects that consume a `Signal` and return a new `Signal`. Many transformers can also operate on a `Collection` or `CollectionMap`; dispatch is handled internally. `Transformer` carries the `key_spec` mechanism for operating on named variables inside `xr.Dataset`-backed signals; `Calculator` adds data validation and shared calculation behavior.

**`Schema` / `DatasetSchema`** ([schema.py](../src/k_onda/central/schema.py))
Dimension metadata that travels with signals through the pipeline without
materializing data.

**`Locus` / `LocusSet`** ([loci/core.py](../src/k_onda/loci/core.py))
Metadata used by `select` to locate a portion of the data. Examples include
markers, events, intervals, epochs, frequency bands, and sets of those objects.

**Compilation And Graph Rewriting** ([signals/core.py](../src/k_onda/signals/core.py), [traversal.py](../src/k_onda/graph/traversal.py), [selector.py](../src/k_onda/transformers/selector/selector.py))
Calling `.compile()` rebuilds a symbolic pipeline into an execution plan. Source signals are shared roots, while derived nodes are rebuilt so execution can cache
intermediate data without mutating the user's original pipeline. Selection is a special case: user-facing `Selector` nodes record intent, and `SelectionPlanner`
rewrites them into concrete `Slicer` nodes at compile time.

**`Annotation` / `ProvenanceContext`** ([provenance.py](../src/k_onda/provenance/provenance.py))
Records the state of mutable objects, such as `Experiment`, `Subject`,
`Session`, and data components, when they enter the DAG.

#### Containers And Derived Outputs

**`Collection` / `CollectionMap` / `SignalMap`** ([sources/core.py](../src/k_onda/sources/core.py))
Containers for working with multiple signals or identities. `Collection` holds
ordered members and lets transformers broadcast over them. `CollectionMap` maps
labels to collections and is produced by `Collection.group_by()`. `SignalMap`
maps labels to signals and is produced by `Aggregator`.

**`SignalStack`** ([signals/core.py](../src/k_onda/signals/core.py))
Vectorized form of a `Collection`: signals are stacked along a new dimension so
downstream calculations can operate over consistent shapes.

**`FeatureRegistry` / `ExtractFeatures`** ([feature_registry.py](../src/k_onda/transformers/feature_registry.py), [feature_transformers.py](../src/k_onda/transformers/feature_transformers.py))
`FeatureRegistry` is a named catalog of mini-pipelines. Each registered function takes a collection and returns an aggregated signal. `ExtractFeatures` applies registered functions across a map and assembles the results into an `IndexedSignal`.

**Mixin System** ([transformer_mixins.py](../src/k_onda/transformers/transformer_mixins.py), [select_mixin.py](../src/k_onda/transformers/selector/select_mixin.py))
`CalculateMixin`, `SelectMixin`, `UnstackMixin`, `IntersectionMixin`, and related mixins compose the fluent API onto signals, collections, and stacks.

**Recipes** ([recipes/core.py](../src/k_onda/transformers/recipes/core.py))
Longer pipelines encoded as named, reusable workflows.

### Data Representation

- `xr.DataArray` and `xr.Dataset` throughout, with `pint` units via `pint_xarray`
- Provenance tracked as graph structure via [graph/traversal.py](../src/k_onda/graph/traversal.py)

### Data Source Modules

- [sources/core.py](../src/k_onda/sources/core.py) - generic source, component, identity, collection, and map infrastructure
- [sources/lfp_sources.py](../src/k_onda/sources/lfp_sources.py) - LFP signal sources
- [sources/spike_sources.py](../src/k_onda/sources/spike_sources.py) - spike and neuron sources

### Transformer Modules

- [transformers/](../src/k_onda/transformers/) - spectral, filter, magnitude, feature, mask, selector, and data-shape transformers
- [transformers/__init__.py](../src/k_onda/transformers/__init__.py) exports the current public transformer surface. Individual modules group related transformers by role rather than defining separate user-facing APIs.
