# What K-Onda Should Learn from FieldTrip

*Synthesized from independent reviews by Codex and Claude Code, March 2026*

## The Big Picture

Both reviews arrived at the same core conclusion: K-Onda's architecture (lazy evaluation, symbolic DAG, xarray backbone, pushdown selection with padding, provenance tracking) is already more sophisticated than FieldTrip's. Don't trade any of that away. Where Pynapple taught lessons about interval algebra and spike operations, FieldTrip's lessons are about **spectral depth, statistical rigor, and the discipline of canonical intermediate representations**.

FieldTrip is not a model for K-Onda's architecture. It is a model for two narrower things: how much scientific surface area a mature electrophysiology toolbox eventually needs, and how to package that surface area into a few stable intermediate data contracts. Both reviews converge on a single distillation:

> Build a small number of scientifically meaningful intermediate representations, and let many analyses compose on top of them.

For K-Onda, those representations are likely to be: interval sets, event/epoch-indexed signals, spectral estimates, cross-spectral / connectivity estimates, and statistical result objects.


## High Priority: Domain Operations K-Onda Needs

### 1. A First-Class Spectral Family, Not Just a Spectrogram

**Both reviews flagged this as the single most important FieldTrip-informed addition.**

FieldTrip treats spectral estimation as a family of related methods with explicit tradeoffs — the core insight being that spectral estimation involves a three-way trade between time resolution, frequency resolution, and variance reduction, and users need to control all three.

The methods it provides:
- **mtmfft** (multitaper FFT): DPSS (Slepian) tapers over the whole signal. `tapsmofrq` sets frequency smoothing width in Hz. More tapers → lower variance, coarser resolution.
- **mtmconvol** (multitaper convolution): Sliding window with DPSS tapers for time-frequency analysis. Frequency resolution = 1/window_length.
- **Morlet wavelet**: Gaussian-windowed sine wave, frequency-dependent time-frequency resolution. More cycles → narrower bandwidth but worse time localization.
- **Hilbert transform**: Bandpass-filter then take analytic signal for instantaneous phase/amplitude. Requires careful filter design.

K-Onda currently has a single `Spectrogram` calculator that assumes a time transform, calls `mne.time_frequency.tfr_array_multitaper`, and produces only time-frequency power output. That's a good beginning, but too narrow to support the rest of the roadmap.

**What K-Onda should steal:** Not the individual algorithms (those are well-documented), but the *parameterization philosophy*. FieldTrip exposes taper choice, smoothing/bandwidth, time-window length, cycles or window-per-frequency, and output representation as first-class configuration. K-Onda's spectral Calculator should offer the same explicit control rather than hiding taper choices behind defaults. The practical recommendation is to split the current `Spectrogram` into a small family of calculators sharing a common output contract, or a single `Spectrum` calculator with an explicit `method` parameter. The important part is not the class naming — it's that K-Onda gets a stable spectral layer broad enough to support everything downstream.

**Why it matters:** LFP and EEG analysis live or die on spectral method choices. The roadmap's Phase 1 spectral work (coherence, MRL, etc.) all depend on getting the underlying spectral estimates right. If K-Onda skips this step, it will end up with one-off calculators that duplicate estimation work and diverge in parameter semantics.

### 2. A Canonical Cross-Spectral Representation and Connectivity as a Unified Framework

FieldTrip's strongest design move is not any single connectivity measure. It is the fact that coherence, Granger, PSI, PDC, DTF, and related analyses are treated as operations on a small number of spectral intermediates: Fourier coefficients, power / cross-spectral density, and transfer-function / MVAR-derived representations.

FieldTrip implements connectivity via a single entry point (`ft_connectivityanalysis`) that dispatches to different measures. All of them consume a cross-spectrum or cross-spectral density matrix, meaning they compose naturally with the spectral analysis step. The pipeline is: raw signal → `ft_freqanalysis` (produces cross-spectra) → `ft_connectivityanalysis` (produces connectivity measure).

K-Onda's roadmap lists coherence, cross-correlation, Granger causality, and MRL, but the codebase does not yet define the shared intermediate representation that would let those methods compose cleanly. The `Schema` / `DatasetSchema` layer is promising, but there is no explicit spectral object contract for "this contains Fourier coefficients / this contains cross-spectra."

**What K-Onda should steal:** Design connectivity Calculators as a *family* that shares a common input format. Don't implement each one as a standalone pipeline — make them compose with each other and with the spectral Calculator. Before implementing coherence, Granger, or PSI individually, define the intermediate signal types or schema contracts they consume and produce. Let users compute one expensive spectral estimate and reuse it for multiple downstream measures. Distinguish at least: time-frequency power, Fourier-domain trial/taper output, and cross-spectral output.

### 3. Cluster-Based Permutation Statistics

**Both reviews identified this as the single biggest domain gap between K-Onda and FieldTrip.**

FieldTrip's nonparametric cluster-based permutation machinery is one of the main reasons people keep using it. It is the gold standard for controlling false positives in time-frequency data and is almost entirely absent from Python neuroscience tools.

**The problem it solves:** When you test for condition differences at every time-frequency point simultaneously, you have a massive multiple-comparisons problem. Bonferroni correction is far too conservative (especially when nearby points are correlated). Cluster-based permutation testing solves this without sacrificing sensitivity.

**The algorithm:**
1. Compute a test statistic at each point.
2. Group adjacent significant points into clusters.
3. Compute a cluster-level statistic (typically summed t-values).
4. Under random permutation of condition labels, repeat and collect the maximum cluster statistic.
5. Compare the observed cluster statistics to the null distribution.

This solves the multiple-comparisons problem in one pass and accounts for the spatial/temporal smoothness of the data. It addresses the actual statistical shape of electrophysiology outputs — time, frequency, channel/sensor, trial/subject structure, and massive multiple-comparisons burdens.

**What K-Onda should steal:** A statistical layer starting with permutation-based comparison and cluster-aware correction for structured outputs. This is a Calculator that takes a grouped signal plus a test specification and returns a significance mask. It doesn't exist anywhere in the current K-Onda design. It belongs on the roadmap as foundational Phase 1 work, not a late add-on. It will be needed before K-Onda can produce publication-ready results.

**Design note:** The statistic should integrate cleanly with K-Onda's provenance graph — a permutation test is itself a transformation step that should appear in the DAG with its parameters recorded.

### 4. Artifact Rejection as Interval Composition

**Both reviews noted the striking fit between FieldTrip's artifact model and K-Onda's architecture.**

FieldTrip treats artifact rejection as a structured, configurable pipeline step. Its `ft_artifact_zvalue` is the core method: z-score each channel, sum z-values across channels, detect intervals exceeding a threshold. Specialized wrappers (`ft_artifact_eog`, `ft_artifact_muscle`, `ft_artifact_jump`) set domain-appropriate defaults. The output is a set of bad intervals that feeds into `ft_rejectartifact`, which removes those intervals from the trial definition — not from the data directly.

This maps beautifully to K-Onda's design:
- `ft_artifact_zvalue` → a Calculator that takes a TimeSeriesSignal and returns an `IntervalSet` of bad intervals
- `ft_rejectartifact` → set-difference algebra on `IntervalSet`s (the composable interval algebra already identified as top priority from the Pynapple review)

Artifact rejection *is* interval composition: `valid_trials = all_trials - artifact_intervals`. K-Onda's planned IntervalSet algebra directly enables this. The interval algebra design should ensure it supports subtraction of detected artifact intervals from an existing EpochSet as a first-class use case.


## Medium Priority: Important But Not Blocking

### 5. Dimension Contracts and Schema Validation

Katie's Note: the robots were confused about what is and isn't implemented.  Those methods are no-ops on the base transformer class and are overridden by others.  

FieldTrip data structures carry a `dimord` string that explicitly names the dimension order (e.g., `'chan_freq_time'`). This is crude compared to xarray, but it solves a real problem: functions know what kind of object they are receiving.

K-Onda's `DatasetSchema` is already a richer version of this concept. But two gaps remain. First, K-Onda doesn't yet enforce consistent dimension ordering across operations — `_validate_input` currently returns `True` unconditionally, and `output_class_for_selection()` is unimplemented. Second, dimension sets are stored as Python `set`s, so ordering is not represented in the schema itself.

This matters because FieldTrip's mature analyses assume strong contracts: connectivity expects specific spectral representations, statistics expect specific repetition/channel/frequency/time axes, plotting expects specific parameter fields. The lesson is not to copy `dimord` — it's that **formal data contracts are what make a big analysis toolbox composable**. K-Onda should strengthen schema validation so calculators can state requirements like "requires a selectable `time` axis" or "requires a spectral representation with `frequency`" or "requires a cross-channel representation."

### 6. Trial Metadata as First-Class Grouping Keys

In FieldTrip, extra columns beyond the basic `[begin, end, offset]` in the trial definition matrix become `data.trialinfo` — a numeric matrix where each row is a trial and columns are condition codes, response latencies, correct/incorrect, etc. This travels with the data through all operations. The mechanism is primitive, but it captures a crucial truth: condition metadata must survive long enough to drive the analysis.

K-Onda already has `conditions` dicts on intervals and loci, subject-level condition storage, and `Collection.group_by(...)`. The FieldTrip pattern suggests one refinement: conditions should be **structured and queryable** — something like `select(correct=True)` on an interval dimension — not just metadata that happens to be attached. The xarray-native version of this is: interval/event conditions become coordinates on the interval dimension, selection and grouping operate on those coordinates, and aggregation uses those coordinates instead of a detached side table. This connects directly to the multilevel aggregation API already on the roadmap.

### 7. Explicit Re-Epoching / Trial Redefinition

FieldTrip's `ft_redefinetrial` lets users reslice existing data: convert stimulus-locked epochs to response-locked epochs, trim time windows, or resegment into shorter fragments. The underlying insight is that **epoch boundaries are not fixed at load time** — researchers routinely need to re-reference the time axis or change what counts as a trial after seeing the data.

K-Onda's Selector already does the mechanical part. What's missing is a user-facing abstraction for re-referencing the time axis relative to a new event. This is closely related to the peri-event alignment concept from the Pynapple review. FieldTrip's experience confirms this is frequently needed and should be in the core API.

### 8. Forward-Backward Filter Convention (Filtfilt)

FieldTrip defaults to zero-phase filtering via forward-backward filtering (`filtfilt`), which avoids phase distortion. It also handles edge-padding explicitly: filters add transient artifacts at trial edges, and FieldTrip pads trials before filtering, then trims.

K-Onda's Filter Calculator and the Selector's pushdown `padlen` mechanism already address the padding issue — which is sophisticated and ahead of most tools. But the choice between causal vs. zero-phase filtering should be a first-class parameter in the Filter Calculator, documented explicitly, because it affects phase-based analyses (coherence, PLV, MRL). These choices are scientific decisions, not implementation details.


## Medium Priority: Design Patterns

### 9. Separating Numeric Kernels from Graph/Bookkeeping

FieldTrip has a consistent pattern: high-level functions handle bookkeeping, cfg validation, provenance, and dispatch; lower-level numerical functions (in `specest/`, `connectivity/`) do the actual math. The algorithm can be tested and reused independently.

K-Onda's Transformer/Calculator split already embodies this. The lesson is to keep it clean as the codebase grows: the Calculator is the high-level dispatch layer (handles schema validation, DAG wiring, input/output typing); the `_apply` method is where the algorithm lives, and it should be as pure as possible. For spectral, connectivity, and statistics work especially, explicitly designing two layers — pure array-level kernels and graph-aware calculator wrappers — will make K-Onda easier to test, easier to benchmark, and easier to expand.

### 10. Reproducibility Export

FieldTrip's `cfg` chain preserves the configuration of every function call in `data.cfg`. Its `reproducescript` feature exports the entire analysis as runnable code. This serves a real user need: researchers want to send a colleague a recipe, not a Python object.

K-Onda's generation graph is structurally superior (traversable, machine-readable, graph-theoretic). The gap is export/serialization — there is not yet a user-facing "export this analysis recipe" path. When this lands, it should look like: DAG to YAML/JSON pipeline spec, DAG to linear recipe report, DAG to compact methods-section summary. K-Onda should solve it in a more structured way than cfg replay.


## Current K-Onda Gaps That Matter Most in This Comparison

These are the current implementation gaps that most directly limit K-Onda's ability to absorb FieldTrip's right lessons:

**Selection semantics are mid-refactor.** `SelectMixin` still has unimplemented string selection, unimplemented `MarkerSet` handling, hard-coded default mode, and fragile kwarg parsing. Any FieldTrip-inspired interval or trial logic should be designed against the intended selector shape, not today's temporary edges.

**Interval collections exist, but algebra does not.** This is the main blocker for artifact workflows, trial composition, and more flexible re-epoching.

**Spectral representation is too narrow for connectivity/statistics work.** Without a canonical Fourier/CSD layer, the roadmap risks accumulating isolated calculators.

**Schema validation is not yet doing enough real work.** FieldTrip's success comes partly from being strict about what each function expects. K-Onda has the right ingredients but not the full enforcement yet.

**Aggregation is entity-level before it is interval-level.** The current `Aggregator` and `CollectionMap` infrastructure is useful but is not yet the same thing as grouping within a signal over event/trial dimensions.


## What NOT to Copy

- **Eager computation.** K-Onda's lazy DAG is more capable. FieldTrip's inability to scale to large recordings or parameter sweeps is a direct consequence of eager evaluation.
- **cfg as a dumping ground.** FieldTrip's unbounded cfg struct accumulates deprecated options, silent misconfigurations, and unclear validation. K-Onda's explicit typed Calculator constructors are cleaner.
- **Trial-centric rigidity.** FieldTrip assumes data is trial-segmented early, making it awkward to work with continuous signals or change epoch definitions later. K-Onda's Selector + Locus system is more flexible — keep the epoch definition late and symbolic.
- **MATLAB-struct metadata conventions.** xarray coordinates + schema are better. Don't model trial info as a numeric matrix.
- **MATLAB-style cell arrays for ragged trials.** K-Onda can use xarray with an explicit `intervals` dimension and handle ragged trials with masking or padding.
- **Source reconstruction complexity.** FieldTrip's source analysis (beamforming, dipole fitting, head models) assumes EEG/MEG sensor geometry. K-Onda targets LFP/spike data where this is not relevant.
- **FieldTrip's breadth indiscriminately.** Sensor geometry, large MEG/EEG plotting subsystems, and similar specializations are not where K-Onda should focus.


## Where K-Onda Is Already Ahead

- **Lazy evaluation.** FieldTrip computes everything eagerly; K-Onda defers computation until `.data` is accessed. This is a scalability and composition advantage FieldTrip will never have.
- **Structured provenance.** K-Onda's generation graph is traversable and machine-readable; FieldTrip's cfg chain is textual and human-readable only.
- **Pushdown Selector with padding.** K-Onda's Selector walks the DAG to push windowing as close to the data source as possible, accumulating filter padding along the way. FieldTrip handles this ad hoc per function. Both reviews confirmed this is one of the strongest parts of K-Onda's current design.
- **Unit-aware computation.** pint + xarray gives automatic unit tracking through transformations. FieldTrip has no equivalent.
- **Fluent API.** `signal.filter(...).normalize(...).select(...)` is more readable and composable than cfg-driven function calls.
- **Locus algebra foundation.** K-Onda's Interval/Marker/Epoch/IntervalSet hierarchy is already more principled than FieldTrip's `trl` matrix. The planned set algebra will complete this advantage.
- **Conditions as dicts on intervals/events.** Already the right generalization of FieldTrip's `trialinfo` pattern.


## Suggested Build Order (FieldTrip-Informed Additions to the Roadmap)

Both reviews converge on roughly the same sequence. These are new items or priority upgrades, in the context of the existing roadmap:

1. **Finish interval collections enough to support real artifact and epoch composition** — the foundation for artifact workflows, trial composition, and flexible re-epoching
2. **Define a canonical spectral representation** — the fulcrum for the most roadmap items; if K-Onda does only one major FieldTrip-inspired thing soon, this is it
3. **Refactor spectral estimation into a small method family** — explicit method selection and parameterization
4. **Build connectivity calculators on top of the spectral representation, not beside it** — shared cross-spectral input format so users compute one expensive estimate and reuse it
5. **Explicit filter phase convention** (zero-phase vs causal) in Filter Calculator — small change, high scientific importance
6. **Strengthen schema / datatype validation** — make calculators validate the presence and role of dimensions, not just their names
7. **Cluster-based permutation testing** — essential for publication-ready output; depends on time-frequency signals and condition metadata
8. **Artifact-detection Calculator** (z-score based) returning an IntervalSet — depends on IntervalSet set algebra
9. **Pipeline export / reproducescript equivalent** — YAML/JSON serialization of the DAG; medium-term, but FieldTrip's experience confirms real demand

Items 1–5 land in Phase 1 of the roadmap. Items 6–8 fill in the gaps FieldTrip highlights for producing real analytical results. Item 9 is longer-term infrastructure.


## Official FieldTrip Sources Consulted

- https://www.fieldtriptoolbox.org/about/
- https://www.fieldtriptoolbox.org/development/datastructure/
- https://www.fieldtriptoolbox.org/faq/development/datatype/
- https://www.fieldtriptoolbox.org/development/module/specest/
- https://www.fieldtriptoolbox.org/development/module/preproc/
- https://www.fieldtriptoolbox.org/faq/spectral/datatype_freq/
- https://www.fieldtriptoolbox.org/tutorial/sensor/timefrequency/
- https://www.fieldtriptoolbox.org/tutorial/sensor/timefrequencyanalysis/
- https://www.fieldtriptoolbox.org/tutorial/connectivity/coherence/
- https://www.fieldtriptoolbox.org/tutorial/connectivity/connectivity_sensor_source/
- https://www.fieldtriptoolbox.org/tutorial/stats/cluster_permutation_timelock/
- https://www.fieldtriptoolbox.org/tutorial/preproc/automatic_artifact_rejection/
- https://www.fieldtriptoolbox.org/faq/preproc/artifact/preproc_filtertypes/
- https://www.fieldtriptoolbox.org/faq/preproc/artifact/preproc_padding/
- https://www.fieldtriptoolbox.org/faq/preproc/events/trialinfo/
- https://www.fieldtriptoolbox.org/faq/preproc/events/eventsversustrials/
- https://www.fieldtriptoolbox.org/example/other/reproducescript/
