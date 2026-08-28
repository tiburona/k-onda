from k_onda.central import type_registry as tr
from ..classifier_calculators import SklearnKMeans


def classify_neurons(
        neuron_collection, 
        spec=None,
        func=None,
        order=None,
        labels=None,
        sort_by=None,
        ):
    if not isinstance(neuron_collection, tr.Collection):
        raise ValueError("`neuron_collection` must be a Collection")
    if not len(neuron_collection):
        raise ValueError("No neurons to classify!")
    if not all([isinstance(neuron, tr.Neuron) for neuron in neuron_collection]):
        raise ValueError("There's a non-neuron in `neuron_collection`.")
    
    stacked_signals = neuron_collection.stack_signals(dim="spike")

    if stacked_signals.data_schema.has_dim("electrode"):
        stacked_signals = stacked_signals.reduce(key="waveforms", dim="electrode", method="mean")



    classified_neurons = (
        stacked_signals
        .median_filter(key="waveforms", kernel_sizes={"sample": 5})
        .unstack_signals()
        .extract_features("fwhm", "firing_rate", group_by="neuron")
        .normalize(method="zscore", dim="index")
        .kmeans(2, implementation=SklearnKMeans(random_state=0))
        .classify(
            "neuron_type", 
            spec=spec,
            func=func,
            order=order,
            labels=labels,
            sort_by=sort_by
            )
    )

    return classified_neurons


classification_registry = {"classify_neurons": classify_neurons}
