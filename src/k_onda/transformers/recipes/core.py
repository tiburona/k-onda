from k_onda.central import type_registry


def classify_neurons(neuron_collection, label_spec):
    if not isinstance(neuron_collection, type_registry.Collection):
        raise ValueError("`neuron_collection` must be a Collection")
    if not len(neuron_collection):
        raise ValueError("No neurons to classify!")
    if not all([isinstance(neuron, type_registry.Neuron) for neuron in neuron_collection]):
        raise ValueError("There's a non-neuron in `neuron_collection`.")

    classified_neurons = (
        neuron_collection.stack_signals(dim="spikes")
        .reduce(key="waveforms", dim="electrodes", method="mean")
        .median_filter(key="waveforms", kernel_sizes={"samples": 5})
        .unstack_signals()
        .extract_features("fwhm", "firing_rate", group_by="neuron")
        .normalize(method="zscore", dim="index")
        .kmeans(n_clusters=2, random_state=0)
        .classify("neuron_type", label_spec=label_spec)
    )

    return classified_neurons


classification_registry = {"classify_neurons": classify_neurons}
