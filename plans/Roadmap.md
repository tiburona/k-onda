# K-Onda Roadmap

## Phase -1 - What's Done

- Pipelines to preprocess LFP data and compute a spectrogram from a single epoch established immutable `Signal`/pure-at-execution `Transformer` architecture.
- `Caclulator`s include (not exhaustive): `ReduceDims`, `Histogram`, `FWHM`, `Spectrogram`, `Filter`, `KMeans`, `Normalize`, `Rate`
- Every signal has DAG-based provenance.  
- `StackSignals` and `UnstackSignals` allow vectorized computation.
- Methods on signals that call transformers are the basis of a fluent API
- `Selector` can select data on multiple dimensions.  It can optionally walk a signal's provenance graph to enable pushdown selection.
- `Intersection` and `ApplyMask` calculators form the basis for masking based on data quality.
- A pipeline to categorize neurons on the basis of extracted features is complete
- A system of `Annotation`s allows changes to mutable entities (like neurons that are categorized) while communicating the context when a signal enters the DAG.


## Phase 0 - To a Minimal Demo
Produce a PSTH plot from our own data.

### 1. Multi-Select  (Currently in Progress)

- Refactor Selector to be able to select over multiple intervals at once 
- Create classes to better represent windows and selections: `Marker`, `Interval`, `EpochSignal`, etc.
- A tentative [implmentation plan](event_interval_multiselect_summary.md) exists.
- Begin writing tests, since Selector is a prime candidate for unit tests to prevent regression.

### 2. Multilevel Aggregation API

- An Aggregator calculator exists, but an API must be developed for successive aggregation over Events, Neurons, Animals, etc., grouped by conditions.  
- Condition metadata does not yet exist and must be added for grouping.

---

### 3. Plotting

- The fluent API must be extended to take an grouped, aggregated signal as an input and output a bar plot.  
- To produce a minimal PSTH, the API also needs methods for labels, a legend, color setting, and the stimulus marker.

---

## Phase 1 — Electrophysiology Foundation  
Make K-Onda a reliable and user-friendly electrophysiology tool for our lab and others.

### 4. Reestablish the Calculation Functionality of the Legacy Version

- More filter types
- MRL
- Autocorrelation  
- Cross-correlation 
- Coherence
- Amplitude
- Granger causality  

---

### 5. Reestablish the Tabular Output Functionality 

- The legacy package delivered outputs as CSV.  Recreate that format, 
- Also consider formats that allow rich metadata to read provenance back into K-Onda.

---

### 6. Extend the Minimal Plotting Functionality 

- The legacy package built on matplotlib. Evaluate this decision.  Plotly?
- The legacy package made heat maps, bar plots, line plots, scatter plots, and polar histograms. Recreate these.
- Reimplement the legacy project's concept of a plot layer.
- Reimplement the legacy project's capacity for nested plots. Add: gridspec-like functionality for more customization. 
- Mechanism to attach statistical significance annotations to plots.

---
### 7. Testing (continued)
- End-to-end tests for major pipelines (spike, power, mrl, etc.) a
- “Golden-file tests” for plots.
- Evaluate further targets for unit tests 

---

### 8. UI/Input

- Build YAML/JSON specification for pipelines
- May need to reimplement something like the legacy projects Runner class.
- Add input validation with informative errors.  


### 9 User-Facing Documentation 
Several complementary modes:
- Recipes: on the line between source code and worked example.  An example is the current informal neuron classification smoke test.  That can be migrated to a recipes.
- Tutorials  
- Encyclopedic-style reference
- Where relevant, scientific documentation (i.e., where K-Onda is opinionated in ways scientists would want to understand and report). Hopefully the new version is less opinionated, but the recipes may be.  

---

### 10. Provenance
- For plots and tabular output, build provenance metadata that can be loaded back into K-Onda, either to reconstruct a pipeline or signal or to pretty print a history.  

---

### 11. Packaging & Command-Line Interface
- Publish to TestPyPI, then PyPI once stable  
- Add a minimal CLI (e.g., `k_onda` or similar) that can:
  - Run an analysis from a config file (`k_onda run --config config.yaml`)  
  - Optionally initialize an analysis repo and add a flag to run-and-commit results  
  - Run diagnostic commands like filter visualization and experimental-design plots
  - Run neuron categorization as a standalone tool 

---

## Phase 2 — Extensibility Framework  
Build the pieces that allow K-Onda to grow.

### 12. Modernization/Improvement of Developer Tooling 
- Install a linter with a commit hook  
- Consider requiring type hints (at least in new/critical modules)  

---

### 13. Developer-Facing Documentation
- Write a short “Contributing” guide (how to run tests, style expectations, where things live)  
- Mark a few “good first issues” for new contributors  
- Add a high-level architecture overview 
- Standardize practices for and expand docstrings

---

### 14. Performance & Resource Use
- Introduce dask-backed arrays for parallelization
- Research whether it is possible to autodect pipelines that could speed up from vectorization. (?)

---

### 15. File Inputs 
- Collect set of standard input formats for electrophys data and add support.  

---

### 16. Plotting and Figure Infrastructure (Phase 2 extensions)  
- Identify electrophysiology-standard plots not yet supported
- Determine user demand for interactive plots, and implement them if its high. 


### 17. Rationalize User Input Process (Phase 2 extensions)
- Add simple GUI (e.g., dropdown-driven pipeline builder, SPM-style)


---

## Phase 3 — Long-Term Ambitious Projects  
Aspirational goals.

### 18. Wrap common curation tools
- Kilosort and Phy for example should be able to be executed from K-Onda and decisison made during manual curation should be captured as graph nodes.
- Identify other similar tools.  

### 19. Statistical Environment & Standardized Outputs
- Embed a version-locked Python/R environment for statistical analyses  
- Return standardized JSON summaries for automated plot annotations  

---

### 20. Plugin Ecosystem
- Formalize interfaces so external labs can contribute new transformers or plot types  
- Define a governance model for reviewing and merging popular plugins  

---

### 21. GUI for Publication-Ready Analysis
- Cross-platform GUI for assembling figures  
- Automatic embedding of provenance  
- Drag-and-drop layout editing  
- Live previews and stat-annotation overlays 

---

### 22. API for Programmer-Scientists
- Define and document a stable public API for K-Onda objects and methods so they can be used in external scripts and notebooks.
- Optionally embed an IPython/Jupyter-style notebook view in K-Onda so experienced Python users can explore data while still keeping analysis history in one place.

---

### 23. Multi-user server mode: Auth, Roles (RBAC), and Remote Execution
- Support authentication (distinct user identities) and authorization (role-based permissions).
- Support a server/worker deployment: run on a host machine; connect from a client.
- Add a security review focused on user-supplied configs/plugins and any user-injectable code paths.

---

### 24. AI-Assisted Statistical and Analytical Guidance
- Suggest appropriate analyses  
- Provide retrieval-augmented explanations referencing the user’s data  
