from collections import defaultdict
from copy import deepcopy

from ..core import Transformer
from k_onda.central import type_registry, DimBounds
from k_onda.graph import list_nodes, rebuild_tree, walk_graph

from .specification import SpecifySelection
from .slicer import SliceSelection

class PlanSelection(Transformer):
    @property
    def fixed_output_class(self):
        return type_registry.SelectorSignal

    def _call_on_signal(self, signal, key_spec=None):

        selector_nodes = [
            node for node in list_nodes(signal) if hasattr(node, "transformer")
            and isinstance(node.transformer, SpecifySelection)
        ]

        leaf = signal

        if len(selector_nodes):
            padlen_accumulator = self._accumulate_padlen(leaf)
            leaf = self._build_slice_plan(leaf, selector_nodes, padlen_accumulator)
        return leaf

    def _accumulate_padlen(self, leaf):

        def merge_state(node, previous_padlen, incoming_padlen):
            merged = deepcopy(previous_padlen)
            return merged.cover_merge(incoming_padlen)

        def step(node, accumulated_padlen, _):
            node_padlen = getattr(node.transform, "padlen", None)
            if node_padlen:
                return accumulated_padlen + node_padlen
            return accumulated_padlen
        
        accumulated_padlen = walk_graph(leaf, DimBounds(), step=step, merge_state=merge_state)

        return accumulated_padlen
    
    def _build_slice_plan(self, leaf, selector_nodes, accumulated_padlen):

        pushdown_selector_nodes = []
        local_selector_nodes = []

        for s in selector_nodes:
            if s.transformer.mode == "pushdown":
                pushdown_selector_nodes.append(s)
            else:
                local_selector_nodes.append(s)

        slicer_plan = defaultdict(list)
       
        if len(pushdown_selector_nodes):
            self._make_pushdown_slicers(
                pushdown_selector_nodes, accumulated_padlen, slicer_plan
            )

        if len(local_selector_nodes):
            self._make_local_slicers(
                local_selector_nodes, accumulated_padlen, slicer_plan 
            )

        self._make_trim_slicers(
            leaf, pushdown_selector_nodes, local_selector_nodes, slicer_plan
            )

        leaf = self._rebuild_tree(leaf, slicer_plan) 

        return leaf

    def _make_local_slicers(
        self, local_selector_nodes, accumulated_padlen, slicer_plan
        ):

        for ls_node in local_selector_nodes:
            padlen = accumulated_padlen[id(ls_node)]
            self._make_slicer(ls_node, ls_node, slicer_plan, padlen=padlen)

    def _make_pushdown_slicers(
            self, selector_nodes, accumulated_padlen, slicer_plan
            ):
        # for each pushdown selector, walk the graph from the selector node
        # carrying the set of dims still looking for placement

        for ps_node in selector_nodes:

            def _check_and_make(node, value, _):
                return self._check_and_make(
                    node,
                    value,
                    ps_node,
                    accumulated_padlen,
                    slicer_plan,
                )

            live_dims = set(ps_node.transformer.locus.dim_bounds)

            walk_graph(ps_node, live_dims, step=_check_and_make)

    def _check_and_make(
        self,
        node,
        live_dims,
        ps_node,
        accumulated_padlen,
        slicer_plan,
    ):

        selectable_dims = node.data_schema.selectable

        def dim_in_inputs(node, dim):
            return any(dim in inp.data_schema.selectable for inp in node.inputs)

        if not live_dims:
            return live_dims
        next_live_dims = set(live_dims)
        for dim in selectable_dims:
            if dim not in next_live_dims:
                continue
            elif not node.is_source and dim_in_inputs(node, dim):
                continue
            elif next_live_dims:
                self._make_slicer(node, ps_node, slicer_plan, accumulated_padlen[id(node)])
                next_live_dims.remove(dim)
            else:
                break

        return next_live_dims

    def _make_trim_slicers(
            self, 
            leaf, 
            pushdown_selector_nodes, 
            local_selector_nodes, 
            slicer_plan
            ):
        # For every pushdown selector place a slicer with the selector's original
        # bounds

        window_nodes = [
            node for node in pushdown_selector_nodes + local_selector_nodes
            if node.transformer.window is not None
            ]
        
        for ps_node in pushdown_selector_nodes:
            trim_bounds = ps_node.transformer.locus.dim_bounds
            for window_node in window_nodes:
                # Detect if the two nodes are attempting to select on different coords
                # on the same dim, where one has a window and one is a pushdown node.  
                # That is a tricky situation we can't yet handle.
                if ps_node.transformer.locus.dim != window_node.transformer.locus.dim:
                    common_metadim = ps_node.data_schema.get_common_metadim(
                        ps_node.transformer.locus.dim, 
                        window_node.data_schema, 
                        window_node.transformer.locus.dim
                        )
                    if common_metadim:
                        raise NotImplementedError(
                            f"You can't yet select on two coords over dim {common_metadim}" 
                            " if they're not the same coord.")
                # Make sure we don't trim more narrowly than the window.
                else:
                    trim_bounds = deepcopy(trim_bounds)
                    trim_bounds.cover(window_node.transformer.locus.dim_bounds)

            self._make_slicer(
                leaf, ps_node, slicer_plan, is_trim=True, trim_bounds=trim_bounds
            )
        
    def _make_slicer(
        self,
        node,
        selector_signal,
        slicer_plan,
        padlen=None,
        is_trim=False,
        trim_bounds=None
    ):
        selector = selector_signal.transformer
        new_dim = selector.new_dim
        window = selector.window if not is_trim else None
        slicer = SliceSelection(
            selector.mode,
            selector.locus,
            new_dim,
            window,
            padlen=padlen,
            is_trim=is_trim,
            trim_bounds=trim_bounds,
            ragged=selector.ragged
        )

        slicer_plan[id(node)].append(slicer)
    
        return slicer
    
    def _rebuild_tree(self, leaf, slicer_plan):

        def insert_slicers(original, rebuilt):
            slicers = slicer_plan.get(id(original))
            if not slicers:
                return rebuilt
            for slicer in slicers:
                rebuilt = slicer(rebuilt)
            return rebuilt
        
        new_leaf = rebuild_tree(leaf, rebuild_node=insert_slicers)

        return new_leaf


