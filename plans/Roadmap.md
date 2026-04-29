# K-Onda Roadmap

## Phase -1 - What's Done

- Pipelines to preprocess LFP data and compute a spectrogram from a single epoch established immutable `Signal`/pure-at-execution `Transformer` architecture.
- `Caclulator`s include (not exhaustive): `ReduceDims`, `Histogram`, `FWHM`, `Spectrogram`, `Filter`, `KMeans`, `Normalize`, `Rate`
- Every signal has DAG-based provenance. 
- Data `Schema`s propogate information about the signal forward.  
- `StackSignals` and `UnstackSignals` allow vectorized computation.
- Mixin Methods on signals that call transformers are the basis of a fluent API
- `Selector` can optionally walk a signal's provenance graph to enable pushdown selection. Validity propagates automatically through output signals.  
- Selector can select over multiple intervals at once 
- `Locus` classes can flexibly represent selections (e.g. `Epoch`, `IntervalSet`, etc.)
- Users can create dimensions with arbitrary names when selecting sets; these become dimensions on the xarray data.  
- `Intersection` and `ApplyMask` calculators form the basis for masking based on data quality.
- A pipeline to categorize neurons on the basis of extracted features is complete
- A system of `Annotation`s allows changes to mutable entities (like neurons that are categorized) while communicating the context when a signal enters the DAG.



## Phase 0 - To a Minimal Demo
Produce a PSTH plot from our own data.

### 1. Multi-Select  (Currently in Progress)

- Mostly done.  
- A tentative [implmentation plan](event_interval_multiselect_summary.md) exists.  Check implementation 
  plan against current reality.
- Begin writing at least a few tests before moving on to Aggregation.

### 2. Multilevel Aggregation API

- An Aggregator calculator exists, but an API must be developed for successive aggregation over Events, Neurons, Animals, etc., grouped by conditions.  
- Make sure that condition metadata propagates to entities lower in the data hierarchy as dimensions on data.

---

### 3. Plotting

- The fluent API must be extended to take an grouped, aggregated signal as an input and output a bar plot.  
- To produce a minimal PSTH, the API also needs methods for labels, a legend, color setting, and the stimulus marker.

---

## Phase 1 — Electrophysiology Foundation  
Make K-Onda a reliable and user-friendly electrophysiology tool for our lab and others.

### 4. Retire Technical Debt/Write Tests

- Review codebase for TODOs.  For instance: the hard disk cache/eager/lazy distinction is promised in comments but not implemented.
- End-to-end tests for major pipelines (spike, power, mrl, etc.) a
- “Golden-file tests” for plots.
- Evaluate further targets for unit tests 

### 5. Expansion of Select Functionality

- Only letting selectors select one dim was a deliberate regression/simplification, but multiple dim 
  support should be added.
- Right now `where` (a method on the `SelectMixin`) only tests for equality of conditions. Expand to: negation, multi-value, ranges on numeric conditions (intensity > 60), and boolean combinations. 
- Develop an ergonomic API for "this event should be normalized relative to its own baseline window."
- Figure out how to handle ragged epochs.

---

### 6. Marker/Interval Set Algebra and Other Functionality

- Union, intersection, difference, etc.
- Should calculators *return* IntervalSets, like Z-score based good intervals versus artifacts? 
- Design question: currently you can currently create a ValidityMask and get its Intersection with a signal, but is this functionality more elegantly handled by IntervalSets such that this could be removed?  Needs thinking about.

---

### 7. Expand validation

- Current DataSchemas and DatasetSchemas are very light weight; expand them.
- Also work on validating user input. 

### 8. Reestablish the Calculation Functionality of the Legacy Version

- Create spectral signal types beyond power (e.g.Fourier-domain trial/taper output and cross-spectral output) and make calculators that produce them.
- Build connectivity calculators (MRL, Coherence, Granger Causality, Correlation)
- Other calculator types: expand Filter (including zero-phase v causal), add Amplitude

---

### 9. Reestablish the Tabular Output Functionality 

- The legacy package delivered outputs as CSV.  Recreate that format, 
- Also consider formats that allow rich metadata to read provenance back into K-Onda.

---

### 10. Extend the Minimal Plotting Functionality 

- The legacy package built on matplotlib. Evaluate this decision.  Plotly?
- The legacy package made heat maps, bar plots, line plots, scatter plots, and polar histograms. Recreate these.
- Reimplement the legacy project's concept of a plot layer.
- Reimplement the legacy project's capacity for nested plots. Add: gridspec-like functionality for more customization. 
- Mechanism to attach statistical significance annotations to plots.

---

### 11. UI/Input

- Build YAML/JSON specification for pipelines
- May need to reimplement something like the legacy project's Runner class.
- Add input validation with informative errors.  

---

### 12. User-Facing Documentation 
Several complementary modes:
- Recipes: on the line between source code and worked example.  An example is the current informal neuron classification smoke test.  That can be migrated to a recipes.
- Tutorials  
- Encyclopedic-style reference
- Where relevant, scientific documentation (i.e., where K-Onda is opinionated in ways scientists would want to understand and report). Hopefully the new version is less opinionated, but the recipes may be.  

---

### 13. Inputs

- Support NWB. 
- Users need means of entering arbitrary data from unknown sources

---

### 14. Packaging & Command-Line Interface
- Publish to TestPyPI, then PyPI once stable  
- Add a minimal CLI (e.g., `k_onda` or similar) that can:
  - Run an analysis from a config file (`k_onda run --config config.yaml`)  
  - Optionally initialize an analysis repo and add a flag to run-and-commit results  
  - Run diagnostic commands like filter visualization and experimental-design plots
  - Run neuron categorization as a standalone tool 

---

## Phase 2 — Extensibility Framework  
Build the pieces that allow K-Onda to grow.

### 15. Provenance
- For plots and tabular output, build provenance metadata that can be loaded back into K-Onda, either to reconstruct a pipeline or signal or to pretty print a history.  

---

### 16. Statistics

- Need calculators for inferential statistics, for example, cluster-based permutation testing.  
- Will also need to define the types of statistical results.  Is this a Signal or something else?

---

### 17. Modernization/Improvement of Developer Tooling 
- Install a linter with a commit hook  
- Consider requiring type hints (at least in new/critical modules)  

---

### 18. Developer-Facing Documentation
- Write a short “Contributing” guide (how to run tests, style expectations, where things live)  
- Mark a few “good first issues” for new contributors  
- Add a high-level architecture overview 
- Standardize practices for and expand docstrings

---

### 19. Performance & Resource Use
- Introduce dask-backed arrays for parallelization
- Research whether it is possible to autodect pipelines that could speed up from vectorization. (?)

---

### 20. Plotting and Figure Infrastructure (Phase 2 extensions)  
- Identify electrophysiology-standard plots not yet supported
- Determine user demand for interactive plots, and implement them if its high. 

---

### 21. Rationalize User Input Process (Phase 2 extensions)
- Add simple GUI (e.g., dropdown-driven pipeline builder, SPM-style)
- Or if we were more ambitious it might be nice to have a GUI with some kind of smart search/autocomplete where based on what kind of signal you could get suggestions for possible next analyses on the bases of a few charaters.

### 22. Wrap common curation tools
- Kilosort and Phy for example should be able to be executed from K-Onda and decisison made during manual curation should be captured as graph nodes.
- Identify other similar tools.  

### 23. Beyond e-phys

- Consider what kind of classes would support researchers with multi-modal data.  

---

## Phase 3 — Long-Term Ambitious Projects  
Aspirational goals.

---

### 24. Statistical Environment & Standardized Outputs
- Embed a version-locked Python/R environment for statistical analyses  
- Return standardized JSON summaries for automated plot annotations  

---

### 25. Plugin Ecosystem
- Formalize interfaces so external labs can contribute new transformers, signal types, or plot types  
- Define a governance model for reviewing and merging popular plugins  

---

### 26. GUI for Publication-Ready Analysis
- Cross-platform GUI for assembling figures  
- Automatic embedding of provenance  
- Drag-and-drop layout editing  
- Live previews and stat-annotation overlays 

---

### 27. Multi-user server mode: Auth, Roles (RBAC), and Remote Execution
- Support authentication (distinct user identities) and authorization (role-based permissions).
- Support a server/worker deployment: run on a host machine; connect from a client.
- Add a security review focused on user-supplied configs/plugins and any user-injectable code paths.

---

### 28. AI-Assisted Statistical and Analytical Guidance
- Suggest appropriate analyses  
- Provide retrieval-augmented explanations referencing the user’s data  
