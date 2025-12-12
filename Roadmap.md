# K-Onda Roadmap

## Phase 1 — Electrophysiology Foundation  
Make K-Onda a reliable and user-friendly electrophysiology tool for our lab and others.

### 1. Fix What’s Actively Broken or Disorganized
- Manual testing/cleanup from pint refactor
- Unit autocorrelation  
- Unit cross-correlation  
- Granger causality  
- Amplitude cross correlation must be refactored to match coherence handling of segments/transforms  
- Decide whether phase–phase MRL returns (currently removed)  
- Move any still-used functions in the deprecated `math_functions` module into the `math` subpackage and remove the old module  

---

### 2. Testing Infrastructure
- Expand end-to-end tests for LFP  
- Add end-to-end tests for Spike  
- Add end-to-end tests for MRL  
- Add unit tests for `aggregates.py`  
- Add “golden-file tests” for plots — generate known-good PNGs and assert they match within a tolerance  

---

### 3. User-Facing Documentation 
Two complementary modes:
- Tutorials / worked examples  
- Encyclopedic-style reference  

What needs documenting:
- Experiment config  
- Calc opts  
- Plot specification schema  
- The combined opts object passed into the runner  

---

### 4. Plotting & Figure Infrastructure
- Add gridspec-style layout objects independent of data partitions (row/col spans; per-cell positional overrides)  
- Add mechanism to attach statistical significance annotations to plots (match stats outputs to K-Onda objects)  

---

### 5. Data Model Stabilization
- Multi-session experiment support  
- Multiple event types and parametrized events (for tuning curves)
- Retire `Group` in favor of Conditions  

Naming refactors:
- `Period` → `Epoch`  
- `Unit` → `Cluster` / `SpikeCluster`  
- `Animal` → `Subject`  

---

### 6. Record-Keeping Improvements
- Autosaved metadata should include the full opts structure (not just `calc_opts`)  
- Save experiment config once per directory  
- Preserve metadata from Kilosort/Phy (e.g., manual curation decisions)  

---

### 7. Rationalize User Input Process (Phase 1)
- Add YAML as a supported input format  
- Simplify or flatten parts of the spec that feel painful  
- Add comprehensive input validation with informative errors  

---

### 8. Minimal Custom / Abstract Modality (for ephys + behavior)
- Implement a basic “Custom” modality that accepts user-provided arrays  
- Keep it very simple for current Safety experiment needs  

---

### 9. Packaging & Command-Line Interface
- Publish to TestPyPI, then PyPI once stable  
- Add a minimal CLI (e.g., `k_onda` or similar) that can:
  - Run an analysis from a config file (`k_onda run --config config.yaml`)  
  - Optionally initialize an analysis repo and add a flag to run-and-commit results  
  - Run diagnostic commands like filter visualization and experimental-design plots
  - Run neuron categorization as a standalone tool 

---

## Phase 2 — Extensibility Framework  
Build the pieces that allow K-Onda to grow.

### 10. Modernization/Improvement of Developer Tooling
- Consider migrating to `pyproject.toml` instead of `setup.py`  
- Evaluate using `uv` or `pixi` instead of plain `pip`  
- Install a linter with a commit hook  
- Consider requiring type hints (at least in new/critical modules)  

---

### 11. Developer-Facing Documentation
- Write a short “Contributing” guide (how to run tests, style expectations, where things live)  
- Mark a few “good first issues” for new contributors  
- Add a high-level architecture overview 
- Standardize practices for and expand docstrings

---

### 12. Performance & Resource Use
- Audit speed and memory usage on all pipelines, looking for bloat
- Provide users with a small set of options to trade off speed vs. memory
- Explore safe opportunities for parallelization (e.g., per animal / per unit / per period) and decide whether to expose a simple “run in parallel” option for users with more cores

---

### 13. File Inputs 
- Think about whether we need NWB support  

---

### 14. Plotting and Figure Infrastructure (Phase 2 extensions)
- Expand `Container` (currently a stub) to support non-data objects (JPEGs, SVGs, etc.)  
- Reintroduce missing plot types from earlier versions: e.g., polar plots  
- Identify any other electrophysiology-standard plots not yet supported  

---

### 15. Allow Further Data/Calculation Transformations
- Users should be able to choose from common math functions (and maybe their own via lambdas) when displaying data. For example, `lag_of_max_corr` should be expressed as normal amp_xcorr + `np.argmax` instead of being a separate bespoke calc.

---

### 16. Rationalize User Input Process (Phase 2 extensions)
- Reconsider natural-language-style parsing for criteria/rules (revive or replace)  
- Add simple GUI (e.g., dropdown-driven spec builder, SPM-style) once specs are stable  

---

### 17. Generalized Modality Architecture
- Make Movement a modality (freezing, position, velocity, rearing, sleep states)  
- Generalize the Phase 1 Custom modality into a more flexible system:
  - User-provided arrays can be per period, per event, or per time bin  
  - Support user-defined transformations (e.g., lambdas)  
- Treat this as a proto-plugin system for non-electrophysiology data  

---

### 18. Generalization of Event Validation / Data Quality Masking
Current event validation works only for LFP/MRL and is power-based.

Users may want to:
- Invalidate more irregular stretches of data (not just contiguous events)
- Invalidate based on raw or filtered data, not just power
- Invalidate spike data
- Provide inputs that invalidate stretches of data based on external criteria (custom masks)

Goals:
- Define a common validation/masking interface that all modalities can use
- Allow user-supplied masks (per time bin, per event, etc.)
- Make validation rules part of the config/spec rather than hard-coded

---

### 19. Further Expand the Data Model
- Make grouping/indexing more flexible:
  - Arbitrary nested or crossed groupings (e.g., collection site, cage)  

---

## Phase 3 — Long-Term Ambitious Projects  
Aspirational goals.

### 20. Statistical Environment & Standardized Outputs
- Embed a version-locked Python/R environment for statistical analyses  
- Return standardized JSON summaries for automated plot annotations  

---

### 21. Next-Generation Specification Model
- Evolve the linear JSON/YAML spec into a DAG  
- Capture branching analyses, caching, and selective recomputation  

---

### 22. Plugin Ecosystem
- Formalize interfaces so external labs can contribute new modalities, processors, or plot types  
- Define a governance model for reviewing and merging popular plugins  

---

### 23. GUI for Publication-Ready Analysis
- Cross-platform GUI for assembling figures  
- Automatic embedding of provenance  
- Drag-and-drop layout editing  
- Live previews and stat-annotation overlays  

---

### 23. AI-Assisted Statistical and Analytical Guidance
- Suggest appropriate analyses  
- Provide retrieval-augmented explanations referencing the user’s data  
