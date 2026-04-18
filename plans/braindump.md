# Braindump

Informal notes / ideas not yet promoted to real plans.

---

## Rename `Signal` → `Node` (and eventually introduce a real `Node` base class)

### The observation

The class currently called `Signal` is doing two jobs at once:

1. **Graph node** — has `inputs`, a `transform`, caches `.data`, participates
   in traversal/planning.
2. **Signal semantics** — time/frequency/point-process shape, units, schema,
   domain-specific methods like `kmeans`, `classify`.

These live at different levels of abstraction, and right now job #1 is being
done *through* job #2. The name `Signal` reads wrong for what the base class
actually is.

### Where the confusion leaks into the code

- `IndexedSignal._materialize` checks
  `all(isinstance(inp, Signal) for inp in self.inputs)`. What it actually
  means is "are all my inputs graph nodes I can pull data from?" — that's a
  Node question, not a Signal question. With a real `Node` base, this would
  become `isinstance(inp, Node)` and naturally include `CollectionMap`,
  `SignalMap`, etc.
- Some of the "is my input a Collection?" branching in `_materialize` would
  collapse if containers were themselves Nodes with their own `_materialize`.
- `list_nodes` / `walk_tree` currently stop at `CollectionMap` (no `.inputs`),
  which is correct-ish for the outer planner pass but is a tell that
  containers are *almost* nodes.
- **`SelectionPlanner.place_slicer`** — graph surgery is where this hurts
  most. `place_slicer` was creating a raw `Slicer` (a Transformer) and
  wiring it into the node list / `next_node.inputs` as if it *were* a node.
  The fix is to call `slicer(node)` so you get a Signal back, but the fact
  that this is a recurring trap says something: when you're doing graph
  manipulation, you're thinking in nodes and edges, but the code forces you
  to think in "call this transformer on that signal to produce a new signal
  that I then patch into another signal's inputs." The calling convention
  hides the graph when you're building pipelines linearly; it leaks badly
  when you're splicing into the middle of an existing chain.

### Sketch of the eventual hierarchy

```
Node                          # graph/lazy-eval machinery (universal)
├── SignalNode                # electrophysiology signal semantics
│   ├── TimeSeriesSignal
│   ├── TimeFrequencySignal
│   ├── PointProcessSignal
│   ├── BinarySignal
│   ├── IndexedSignal
│   └── SignalStack
├── AssessmentNode            # future: behavioral/assessment domain
└── ...                       # future domains
```

The graph/lazy-evaluation machinery is the universal part; the domain
(electrophysiology, assessments, whatever) is the specialized part. Keeping
those on separate axes means adding a new domain does not require touching
signal code.

### When to actually do this

Not yet. The confusion is currently nominal — `Signal` reads wrong, but
nothing is structurally blocked by it. The refactor becomes worth its cost
when either:

- **(a)** a second kind of node shows up (e.g. AssessmentNode) and the
  graph/caching logic starts getting copy-pasted, or
- **(b)** the container classes (`Collection`, `CollectionMap`, `SignalMap`)
  grow enough node-like behavior that the special cases in `_materialize`
  and `list_nodes` start hurting.

The `CollectionMap`-as-leaf case (the traversal / `_gather_selectors` /
`_materialize` crashes that showed up during multi-select work) is a mild
version of (b) — not painful yet, but a data point.

### Cheap intermediate move

If the *name* specifically starts bugging me before the full refactor is
justified:

- Alias `Node = Signal` in the graph module.
- Start using `Node` in traversal code and in isinstance checks that are
  really about node-ness rather than signal-ness.

Costs nothing, and when the real split happens later, most of the mechanical
renaming work is already done — the Signal subclasses just slot under a real
`Node` base.
