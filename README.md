K-Onda is a Python application for analyzing electrophysiology experiments. It will serve two audiences: researchers who are comfortable programming in Python and researchers who are looking for a higher-level application that still provides them with the transparency and control of custom scripts. It aims to develop into an application that can unify disparate stages of data analysis into a workflow that will create a provenance record as a byproduct of using it.

The original version of K-Onda, now largely deprecated, grew organically out of our lab's analysis needs.  As the project grew, the shortcomings of the underlying data model became clear. This new version, currently in its earliest development stages, starts from scratch.

It operates under the following principles (all of which were sometimes violated by the legacy project):

1. A calculation and its data are separate concerns, and are represented orthogonally, enabling the generalization and reuse of code.
2. Data transformations are atomic and pipelines are composable. K-Onda will provide pre-baked recipes based on common practices in the field, but the user has total control over their own pipelines.
3. K-Onda looks for ways to make the UI friendly and intuitive. The first two interfaces for specification of a pipeline are a fluent API for exploratory use and a YAML/JSON configuration API.  The latter allows reproducible and shareable pipelines, as well as providing an interface to the application for users who do not want to program.  The non-programming specification will have the same flat, linear, "sentence-like" quality as the fluent API, and will extend naturally into a GUI.
4. Data pipelines are represented symbolically first and calculations are only performed when the user finally requests data.  Every transformation makes a record of itself on a provenance graph that is eventually stored on the output, making the artifacts self-documenting, depending on reference to neither the internals of the codebase nor even its official documentation.
5. K-Onda provides mechanisms for expressing the quality and uncertainty of data at each stage of a pipeline. This metadata propagates through transformations alongside the data itself, enabling downstream analyses to weight, filter, or flag results according to flexible standards.
6. Any data with consistent shapes can be vectorized for performance, and the fluent API and configuration spec include methods to make this intuitive and low-effort.
7. Individual data transformations, once configured, act as pure functions at execution time: they take data in and produce data out with no side effects. This design will eventually enable parallelization across channels, trials, or subjects.
8. Different data transformations have different default methods for data access: caching on disk, loading and storing in memory, or recomputing on demand, depending on what is most expensive for the relevant computation.  This enables efficient use of RAM and processor time.
9. Plotting, tabular export, and statistical analysis all derive from a single canonical data representation, preventing divergence.
10. Where feasible, K-Onda looks for data representations that are general, and can extend beyond any particular data format or collection strategy.  It will provide a foundation that can extend even beyond electrophysiology, to other kinds of data in the biological and social sciences.

The original project included a declarative plotting specification that let users compose complex multipanel, multilayer figures. The new version will reimplement and extend this functionality. For more on planned features, see the [roadmap](plans/Roadmap.md).  


Statement on the use of generative AI in this project:

K-Onda is currently the work of a solo developer.  She finds her robot overlords very useful for ideas, feedback, planning, code review, and rubber ducking.  However, it is also her experience that in order to maintain deep, elaborated engagement with the internals of her own project, for anything more complicated than very rote boilerplate, she must write the code herself. 
