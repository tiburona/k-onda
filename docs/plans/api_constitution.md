## API Rules

These rules govern fluent methods and transformer constructors. YAML and JSON are accepted only by configuration APIs (right now, only Experiment), not by individual fluent operations.

1. Use positional arguments for inputs that define the operation.  Modifiers whose meaning is secondary or non-obvious go after * (require them to be passed by keyword).

Examples:

```
signal.scale(0.25, key="power")
signal.reduce("time", method="mean")
signal.threshold(">", 4)
signal.intersection(other_signal, tolerance_decimals=6)
```

2. If an operation is not complex enough to require nesting, configure with the method's parameters directly: 

Examples:

```
signal.scale(0.25)
loci.where(conditions={"treatment": "drug"})
```

3. For operations that are complex enough to potentially require nesting (for now, examples of this are several methods of the plotting API and .classify()), accept either a dictionary specification as a keyword arg, or a flat set of keyword args for the simple case.  Raise if you get the dictionary spec and any other keyword arg that shares the responsibilities of the dictionary spec.  Raise if there are unknown fields in the dictionary spec.  


4. When a fluent method directly applies one transformer, the transformer's
constructor should mirror the fluent method's configuration parameters.
Parameters that control how the transformer is applied, including additional
data inputs, instead mirror parameters to the transformer's `__call__`.

Some fluent methods are planners rather than direct transformer wrappers. They
may normalize public syntax, inspect the planned schema, and compose multiple
transformers or directives. `select()`, collection `mean()`, and the plotting
methods that convert public dictionaries or keyword arguments into plot-spec
records are current examples. Their signatures should describe the public
operation rather than mirror any one internal transformer or directive. These
methods validate their public syntax at the API boundary; the objects they
configure still validate their own normalized configuration. Planning is
permitted here, but numerical calculation and rendering remain in the configured
objects.

Constructor parameters used only for developer-facing dependency injection are
not operation configuration and need not appear in a fluent method. They remain
keyword-only. `Render`'s resolver and renderer collaborators are the current
example.
