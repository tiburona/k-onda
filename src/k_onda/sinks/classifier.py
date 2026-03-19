import json
import yaml

from k_onda.transformers import KMeans, ExtractFeatures
from k_onda.central import ureg, operations
from k_onda.sources import Collection


class Classify:

    def __init__(self, label_name, label_spec=None, label_func=None, spec_format='yaml'):
        self.label_name = label_name
        self.classification_spec = label_spec
        self.label_func = label_func
        self.spec_format = spec_format
        if self.classification_spec is None and self.label_func is None:
            raise ValueError("One of label_spec and label_func must not be None.")

    def _validate_input(self, *inputs):
        from k_onda.signals import IndexedSignal
        for input in inputs:
            if not isinstance(input, IndexedSignal):
                raise ValueError("Input must be of type IndexedSignal.")

    def __call__(self, *inputs):
        self._validate_input(*inputs)

     
        if self.classification_spec and not self.label_func:
            labeled_entities = self._parse_spec(*inputs)
        else:
            labeled_entities = self.label_func(*inputs)
        
        return Collection(labeled_entities)
    
    def _parse_spec(self, *chain):
        
        entities_to_label = chain[0].data.coords['index'].values

        if self.spec_format == 'yaml':
            spec = yaml.safe_load(self.classification_spec)
        elif self.spec_format == 'json':
            spec = json.loads(self.classification_spec)
        else:
            raise ValueError("Unknown specification format for Classfiier")
        
        # spec is a list of dictionaries. each item in the list is a rule
        # each successive rule is allowed to override the next rule

        for i, rule in enumerate(spec):
            spec_type = rule['type']  # threshold | classifier | default
            if spec_type == 'classifier':
                # example classifier specification:
                # feature: 'fwhm'
                # order: 'ascending'  # ascending | descending
                # labels: ['IN', 'PN']  # in order of their value on the selected feature
                feature = rule['feature']
                order = rule.get('order', 'ascending')
                labels = rule['labels']
                
                classification = [node for node in chain if isinstance(node.transformer, KMeans)][0] 
                
                # centers is a n_clusters, n_features numpy array
                centers = classification.data.attrs['kmeans_centers']  
                feature_names = classification.data.attrs['kmeans_feature_names']
                
                # the column ind of the feature that determines the label, e.g. 'fwhm'
                feature_ind = feature_names.index(feature) 
                
                # sort row indexes on the value of the center at the selected feature.
                center_inds, feature_vals = list(range(len(centers))), centers[:, feature_ind]
                sorted_inds_and_vals = sorted(
                    zip(center_inds, feature_vals), key=lambda x: x[1]
                    )
                if order != 'ascending':
                    sorted_inds_and_vals = reversed(sorted_inds_and_vals)
                
                # map the sorted indices to the sorted labels 
                inds_and_labels = {}
                for i, (ind, _) in enumerate(sorted_inds_and_vals): 
                    inds_and_labels[ind] = labels[i]

                # assign labels to the entity in the index
                for i, entity in enumerate(entities_to_label):
                    int_label = int(classification.data.values[i])
                    entity.set_annotation(
                        self.label_name, 
                        inds_and_labels[int_label], 
                        annotator=self,
                        source_signal=classification
                        )

            elif spec_type == 'threshold':
                # example threshold specification

                #     - feature: fwhm
                #       operator: '>'
                #       value: 400
                #       from_computed_features: True
                #       unit: us
                #       label: PN
                feature = rule['feature']
                operator = rule['operator']
                value = rule['value']
                from_computed_features = rule.get('from_computed_features')
                unit = rule.get('unit')
                if unit:
                    value *= ureg(unit)
                label = rule['label'] 

                feature_set = None
                if from_computed_features:
                    # value comes from a previously computed output of ExtractFeatures transformer
                    feature_set = [
                        node for node in chain
                        if isinstance(node.transformer, ExtractFeatures)
                        ][0]
                    feature_vals = feature_set.data.sel(feature=feature)

                else:
                    # default: we'll look for the feature on the entity
                    feature_vals = [
                        getattr(entity, feature) 
                        for entity in entities_to_label
                        ]
                
                test_func = operations[operator]
                for entity, fval in zip(entities_to_label, feature_vals):
                    if test_func(fval, value):
                        setattr(entity, self.label_name, value)
                        entity.set_annotation(self.label_name, label, annotator=self, source_signal=feature_set)
    
            elif spec_type == 'default':
                # example default specification
                # value: PN

                if i != 0:
                    raise ValueError("Default value must come first in classification spec.")
                
                value = rule['value']
                
                for entity in entities_to_label:
                    setattr(entity, self.label_name, value)
                    entity.set_annotation(self.label_name, value, annotator=self, source_signal=None)
                
            else:
                raise ValueError("Unknown classification type")
        
        return entities_to_label

   