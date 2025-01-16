from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from .feature_plotter import FeaturePlotter
from ..plotting_helpers import smart_title_case


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


class CategoryPlotter(FeaturePlotter):
    
    def transform_divisions(self, divisions):
         
        new_divisions = {}
         
        for key in divisions.keys():
            if key == 'data_source':
                new_divisions[divisions['data_source']['type']] = divisions['data_source']
            else:
                new_divisions[key] = divisions[key]

        return new_divisions
    
    def assign_positions(self, divisions, aesthetics, base_position=0, label_prefix=None):
        
        if not label_prefix:
            label_prefix = []


        if not divisions:
            # Base case: no more divisions
            return {}
        
        division = divisions[0]
        remaining_divisions = divisions[1:]

        spacing = aesthetics.get('default', {}).get('spacing', 2)  # Get 'spacing' from division_info
        members = division['members']

        label_to_pos = {}
        position = base_position

        for member in members:
            current_label = deepcopy(label_prefix)
            if isinstance(member, str):
                current_label.append(member)
            else:
                current_label.extend(list(v for v in member.values()))
  
            if remaining_divisions:
                # Recursively assign positions for subcategories
                sub_positions = self.assign_positions(
                    remaining_divisions,
                    aesthetics,
                    base_position=position,
                    label_prefix=current_label
                )
                label_to_pos.update(sub_positions)

                # Update position after processing subcategories
                if sub_positions:
                    last_pos = max(sub_positions.values())
                    position = last_pos + spacing
                else:
                    # No subcategories, increment position by spacing
                    position += spacing
            else:
                # Base case: assign position to the composite label
                label_to_pos[tuple(current_label)] = position
                position += spacing

        return label_to_pos
    
    def get_composite_label(self, spec, row):
        label = []
        divider_types = set([division['divider_type'] for division in spec['divisions']])
        for divider_type in divider_types:
            if isinstance(row[divider_type], str):
                label.append(row[divider_type])
            else:
                label.extend([v for d in row[divider_type] for v in d.values()])
        return tuple(label)
       
    def process_calc(self, calc_config):

        info, spec, aesthetics = (calc_config[k] 
                                  for k in ['info', 'spec', 'asethetics'])
    
        transformed_divisions = deepcopy(spec['divisions'])
        self.label_to_pos = self.assign_positions(transformed_divisions, aesthetics)
        ax = row['cell']
        ax_args = aesthetic_args.get('ax', {})
        self.apply_ax_args(ax, ax_args)

        for row in info:
            composite_label = self.get_composite_label(spec, row)
            position = self.label_to_pos[composite_label]

            aesthetic_args = self.get_aesthetic_args(row, aesthetics)
            marker_args = aesthetic_args.get('marker', {})

            self.cat_width = aesthetic_args.get('cat_width', 1)

            self.plot_markers(ax, position, composite_label, row, marker_args, aesthetic_args=aesthetic_args)
            
        # Set x-ticks and labels
        positions = list(self.label_to_pos.values())
        labels = [smart_title_case(' '.join(label).replace('_', ' ')) 
                  for label in self.label_to_pos.keys()]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

class CategoricalScatterPlotter(CategoryPlotter):

    def plot_markers(self, ax, position, _, row, marker_args, aesthetic_args=None):
        scatter_vals = row[row['attr']]

        # Generate horizontal jitter
        jitter_strength = aesthetic_args.get('max_jitter', self.cat_width/6)
        jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(scatter_vals))
        x_positions = position + jitter
        # Plot with jittered positions
        ax.scatter(x_positions, scatter_vals, **marker_args)
        # Retrieve positions and bar width
        cat_width = aesthetic_args.get('cat_width', 1)
        if 'background_color' in aesthetic_args:
            background_color, alpha = aesthetic_args.pop('background_color')
            ax.axvspan(
                position - cat_width / 4, # TODO cat_width isn't really doing anything now
                position + cat_width / 4,
                facecolor=background_color, alpha=alpha)
    

class CategoricalLinePlotter(CategoryPlotter):

    def plot_markers(self, ax, position, _, row, marker_args, aesthetic_args=None):
        bar_width = self.cat_width
        divisor = aesthetic_args.get('divisor', 2)
        width = bar_width / divisor
        ax.hlines(
            row['mean'],
            position - width / 2,
            position + width / 2,
            **marker_args)

        
class BarPlotter(CategoryPlotter):
   
    def plot_markers(self, ax, position, _, row, marker_args, aesthetic_args=None):
        ax.bar(position, row[row['attr']], **marker_args)