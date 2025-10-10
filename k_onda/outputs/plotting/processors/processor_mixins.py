from copy import deepcopy
from k_onda.utils import recursive_update, smart_title_case, safe_get


class ProcessorMixin:
    pass


class LayerMixin(ProcessorMixin):        

    def init_layers(self):
        layers = deepcopy(self.spec.get('layers', {}) or self.layers)
        if not layers:
            return
        if not self.info_by_division_by_layers:
            self.info_by_division_by_layers = [[] for _ in layers]
        return layers
     
    
    def get_layer_dicts(self, info):
        for i, layer in enumerate(self.layers):

            attr = layer.get('attr', self.spec.get('attr', 'calc'))

            if 'calc_opts' in layer:
                recursive_update(self.calc_opts, layer['calc_opts'])
                self.calc_opts = self.calc_opts

            # if not info.get('data_source'):
            #     data_source = self.experiment
            # else:
            #     data_source = self.get_data_sources(data_object_type = info['data_source'], 
            #                                 identifier=info[info['data_source']])

            new_d = self.copy_info(info)
            new_d.update({
                'layer': i, 
                'attr': attr})
                #attr: getattr(data_source, attr), 
                #data_source.name: data_source.identifier})
            
            self.info_by_division_by_layers[i].append(new_d)
            self.get_calcs(new_d)
    

class AestheticsMixin(ProcessorMixin):
 
    def init_aesthetics(self):
        element_dict = self.spec.get('aesthetics', {})
        if self.parent_processor:
            recursive_update(self.parent_processor.aesthetics, element_dict)
            return self.parent_processor.aesthetics
        else:
            return element_dict


class LegendMixin:

    @property
    def has_colorbar(self):
        return bool(self.colorbar_spec)

    @property
    def colorbar_spec(self):
        return self.spec.get('legend', {}).get('colorbar', {})
            
    @property
    def colorbar_for_each_plot(self):
        return self.has_colorbar and self.colorbar_spec.get('share') in ['each', None]

    @property
    def global_colorbar(self):
        return self.has_colorbar and self.colorbar_spec.get('share') == 'global'



class LabelMixin:

    def get_label(self):
       return self.spec.get('label', {})

    def get_next_label(self):
        if not self.next:
            return None
        else:
            next_spec = list(self.next.values())[0]
            return next_spec.get('label', {})

    def _process_label_text(self, config, data_source):
        """Processes the label text: fills fields and applies smart case."""
        text_template = config.get('text')
        if not text_template:
            return None 

        processed_text = self.fill_fields(text_template, obj=data_source)

        if config.get('smart_case', True):
             processed_text = smart_title_case(processed_text.replace('_', ' '))

        return processed_text
    
    def _get_figure_label_setter(self, position, config):
        """Gets the setter function or fig.text parameters for FIGURE labels."""
        kwargs = config.get('kwargs', {}).copy()
        fig = self.child_layout.label_figure
        setter = None
        is_fig_text = False

        def _create_fig_text_setter(default_x, default_y, alignment_defaults):
            local_kwargs = kwargs.copy()
            x_val = local_kwargs.pop('x', default_x)
            y_val = local_kwargs.pop('y', default_y)
            for key, value in alignment_defaults.items():
                local_kwargs.setdefault(key, value)
            return lambda txt: fig.text(x_val, y_val, txt, **local_kwargs)

        if position == 'title':
            setter = getattr(fig, 'suptitle', None)
        elif position == 'x_bottom':
            setter = getattr(fig, 'supxlabel', None)
        elif position == 'y_left':
            setter = getattr(fig, 'supylabel', None)
        elif position == 'x_top':
            is_fig_text = True
            setter = _create_fig_text_setter(0.5, 0.98, {'ha': 'center', 'va': 'top'})
        elif position == 'y_right':
            is_fig_text = True
            setter = _create_fig_text_setter(0.98, 0.5, {'rotation': 'vertical', 'ha': 'right', 'va': 'center'})
        else:
            print(f"Warning: _get_figure_label_setter called with invalid position: {position}")
            return None

        if setter is None and not is_fig_text:
             print(f"Warning: Could not find figure method for position '{position}'.")
             return None

        return setter

    def _get_ax_label_setter(self, cell, position, config):
        """Gets the setter function for AXES labels based on position.
        Supports 'x_bottom', 'x_top', 'y_left', 'y_right', and 'title'.
        """
        kwargs = config.get('kwargs', {}).copy()
        mapping = dict.fromkeys(['x', 'x_bottom'], ('bottom', cell.set_xlabel, 'xaxis'))
        mapping.update(dict.fromkeys(['x_top'], ('top',    cell.set_xlabel, 'xaxis')))
        mapping.update(dict.fromkeys(['y', 'y_left'], ('left',  cell.set_ylabel, 'yaxis')))
        mapping.update(dict.fromkeys(['y_right'],  ('right', cell.set_ylabel, 'yaxis')))
        mapping.update(dict.fromkeys(['title', 'title_center'], ('center', cell.set_title)))
        mapping.update(dict.fromkeys(['title_left'], ('left', cell.set_title)))  
        mapping.update(dict.fromkeys(['title_right'], ('right', cell.set_title)))  
        
        if position not in mapping:
            print(f"Warning: _get_ax_label_setter called with invalid position: {position}")
            return None
        if position[0:5] == 'title':
            pos, set_label_func = mapping[position] 
            def setter(text, **extra_kwargs):
                cell.set_title(text, loc=pos, **{**kwargs, **extra_kwargs})
        else:
            pos, set_label_func, axis_attr = mapping[position]
            def setter(text, **extra_kwargs):
                getattr(cell, axis_attr).set_label_position(pos)
                set_label_func(text, **{**kwargs, **extra_kwargs})
        return setter
    
    def _get_label_setter_and_kwargs(self, position, config, cell):
        """
        Determines the appropriate label setter based on the config and returns it along with any additional kwargs.
        The configuration must specify the target type under the key 'target' (either 'axes' or 'figure').
        Defaults to 'axes' if not provided.
        """
        target = config.get('target', 'axes')
        if target == 'subfigure':
            setter = self._get_figure_label_setter(position, config)
        else:
            setter = self._get_ax_label_setter(cell, position, config)
        return setter, {}

    
    def set_label(self, cell, updated_info):
        """Sets various labels on the figure or axes based on the spec."""
        label_config = self.get_label() 
        if not label_config:
            return

        needs_data_source = any('{' in conf.get('text', '') 
                                for conf in label_config.values())
        data_source = None
        if needs_data_source:
            ds_type = updated_info.get('data_source')
            ds_identifier = updated_info.get(ds_type) if ds_type else None
            if ds_type and ds_identifier:
                data_source = self.get_data_sources(data_object_type=ds_type,
                                                    identifier=ds_identifier)
      
            else:
                 print("Warning: Label text contains placeholders, "
                 "but no data_source info provided in updated_info.")
              

        # Process each defined label position
        for position, config in label_config.items():
            if position != 'title' and config.get('which', 'all') != 'all':
                axis = position[0]
                absolute = 'absolute' in config['which']
                last = 'last' in config['which']
                if not cell.is_in_extreme_position(axis, last, absolute):
                    continue

            if not isinstance(config, dict):
                print(f"Warning: Invalid configuration for label position '{position}'. Expected a dict.")
                continue

            # You can't set ax labels until you've made the axes, and you don't 
            # until you're at the end of the processor cascade
            if config.get('target', 'axes') == 'axes' and self.next:
                raise ValueError("Cannot set axes labels before the axes are created.")

            processed_text = self._process_label_text(config, data_source)

            label_setter, kwargs = self._get_label_setter_and_kwargs(position, config, cell)

            if label_setter:
                label_setter(processed_text, **kwargs)
            

    def is_condition_met(self, category, member, **_):
        
        return (
            getattr(self, f'selected_{category}', None) == member or
            self.selected_conditions.get(category) == member or
            category == 'period_types' and member in self.selected_period_types
            )


class MarginMixin:

    def calculate_rect(self, margin_spec):
        width = 1 - (margin_spec['right'] + margin_spec['left'])
        height = 1 - (margin_spec['top'] + margin_spec['bottom'])
        rect = (margin_spec['left'], margin_spec['bottom'], width, height)
        return rect
    
    def calculate_margins(self, margin_spec):
        margin_spec = deepcopy(margin_spec)
        for k, v in margin_spec.items():
            if k in ('top', 'right'):
                margin_spec[k] = 1 - v
        return margin_spec
            
