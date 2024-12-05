from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import operations


class NeuronClassifier:
    
    def __init__(self, experiment):
        self.experiment = experiment
        config = self.experiment.exp_info['categorize_neurons']
        self.kmeans = config.get('kmeans')
        self.neuron_types = config['neuron_types_ascending']
        if config.get('neuron_colors'):
            neuron_colors = config['neuron_colors']
        else:
            neuron_colors = plt.cm.get_cmap('viridis', len(self.neuron_types_ascending))
        self.neuron_colors = {t: c for t, c in zip(self.neuron_types, neuron_colors)}
        self.characteristics = config.get('characteristics', ['firing_rate', 'fwhm_microseconds'])
        self.sort_on = config.get('sort_on', 'firing_rate')
        self.plot_characteristics = config.get('plot')
        self.sort_on_ind = self.characteristics.index(self.sort_on) 
        self.cutoffs = config.get('cutoffs', [])
        self.block_ad_hoc_cutoffs_setting = config.get('block_ad_hoc_cutoffs_setting', False)
        self.labels = None
        self.centers = None

    @property
    def units(self):
        return [unit for unit in self.experiment.all_units if unit.category == 'good']

    def classify(self):

        #TODO: add check to make sure all neurons are classified
        saved_calc_exists, categorized_neurons, pickle_path = self.experiment.load(
            self.experiment.construct_path('spike'), ['classified_neurons'])
        if not saved_calc_exists:
            categorized_neurons = self.categorize_neurons(pickle_path)
        self.apply_categorization(categorized_neurons)

    def apply_categorization(self, categorized_neurons):
        for animal in [animal for animal in self.experiment.all_animals if animal.include()]:
            if 'neurons' in animal.initialized:
                continue
            neurons = categorized_neurons[animal.identifier]
            for unit in animal.units['good']:
                neuron_type = neurons[unit.identifier]
                unit.neuron_type = neuron_type
                animal.neurons[neuron_type].append(unit)
            animal.initialized.append('neurons')
   
    def get_input_to_kmeans(self, characteristic):
        scaler = StandardScaler()
        raw = [getattr(unit, characteristic) for unit in self.units]
        scaled = scaler.fit_transform(np.array(raw).reshape(-1, 1))
        return scaled, np.array(raw)  # Return both scaled and original values
    
    def run_kmeans(self):
        X_scaled, X_original = zip(*[self.get_input_to_kmeans(char) 
                                     for char in self.characteristics])
        X_scaled = np.column_stack(X_scaled)
        X_original = np.column_stack(X_original)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=len(self.neuron_types), random_state=0)
        kmeans.fit(X_scaled)
        self.labels = kmeans.labels_
        self.centers = kmeans.cluster_centers_

        # Sort cluster centers based on the specified characteristic and map indices to neuron types
        sorted_indices = np.argsort(self.centers[:, self.sort_on_ind])
        index_to_type = {index: neuron_type 
                            for index, neuron_type in zip(sorted_indices, self.neuron_types)}
        
        return X_original, [index_to_type[label] for label in self.labels]
    
    def apply_cutoffs(self):
        for i, unit in enumerate(self.units):
            for characteristic, operator, value, label in self.cutoffs:
                attr = getattr(unit, characteristic)
                operation = operations[operator]
                if operation(attr, value):
                    unit.neuron_type = label
                    self.labels[i] = label 

    def categorize_neurons(self, pickle_path):

        if self.kmeans:
            self.X_original, self.labels = self.run_kmeans()

            # Assign neuron types based on k-means results and apply cutoffs
            for unit, label in zip(self.units, self.labels):
                unit.neuron_type = label

        else:
            _, self.X_original = zip(*[self.get_input_to_kmeans(char) 
                                     for char in self.characteristics])
            self.labels = [None for _ in self.units]

        done = False

        while not done:
            self.apply_cutoffs()

            if self.plot_characteristics:
                self.plot_neurons()
                happy = input("Are you happy with this categorization? (y/n)")
                if happy == 'y':
                    done = True
                else:
                    if not self.block_ad_hoc_cutoffs_setting:
                        self.get_ad_hoc_cutoffs()
            else:
                done = True
            
        # Prepare results for saving
        to_return = defaultdict(dict)
        for animal in self.experiment.all_animals:
            for unit in animal.all_units:
                to_return[animal.identifier][unit.identifier] = unit.neuron_type

        # Save the results
        self.experiment.save(to_return, pickle_path)
        return to_return
    
    def plot_neurons(self):
        
        color_map = [self.neuron_colors[label] for label in self.labels]

        if len(self.plot_characteristics) == len(self.characteristics):
            orig_data = self.X_original
        else:
            orig_data = np.column_stack(
                [self.get_input_to_kmeans(characteristic, self.units)[1] 
                 for characteristic in self.plot_characteristics])

        # Set up subplots: scatter plot on the left and one box plot for each characteristic
        fig, axes = plt.subplots(1, len(self.characteristics) + 1, 
            figsize=(14 + 6 * (len(self.characteristics) - 1), 6),
            gridspec_kw={'width_ratios': [3] + [1] * len(self.characteristics)})

        # Scatter plot on the left
        if self.X_original.shape[1] == 1:
            y_data = np.random.normal(0, 0.05, size=self.X_original.shape[0])
            y_label = 'Jittered Axis'
        else:
            y_data = self.X_original[:, 1]
            y_label = 'Original ' + self.characteristics[1]

        axes[0].scatter(orig_data[:, 0], y_data, c=color_map, alpha=0.6, edgecolors='w', s=50)
        axes[0].set_xlabel('Original ' + self.characteristics[0])
        axes[0].set_ylabel(y_label)
        axes[0].set_title('Neuron Type Classification')

        # Box plots for each characteristic on the right
        for i, characteristic in enumerate(self.characteristics):
            box_data = [orig_data[np.array(self.labels) == nt][:, i] 
                        if orig_data.shape[1] > 1 else orig_data[np.array(self.labels) == nt]
                        for nt in self.neuron_types]
            
            # Create box plot with specified colors for each neuron type
            box_plot = axes[i + 1].boxplot(box_data, vert=True, patch_artist=True)
            for patch, color in zip(
                box_plot['boxes'], 
                [self.neuron_colors[nt] for nt in self.neuron_types]):
                patch.set_facecolor(color)
            
            axes[i + 1].set_xticks(range(1, len(self.neuron_types) + 1))
            axes[i + 1].set_xticklabels(self.neuron_types)
            axes[i + 1].set_title(f'Mean & Std Dev of {characteristic}')
            axes[i + 1].set_ylabel(characteristic)

        plt.tight_layout()
        plt.show()

    def get_ad_hoc_cutoffs(self):

        characteristic = None

        while characteristic != '0':
            # Command line interaction for threshold setting
            characteristic = input(
                f"Enter the characteristic for which to set a threshold. Characteristics are: {', '.join(self.plot_characteristics)}."
                " Enter 0 if you're done.")
        
            if characteristic != '0':
                operator = input(
                    "Enter the operator to use to make the comparison. "
                    "Options are <, >, <=, >=, ==, !=")
                print(f"You picked {operator}")
                comparison_val = float(input("Enter the comparison value for {}: ".
                                        format(characteristic)))
                print(f"You picked {comparison_val}")
                neuron_type = input("When this comparison is true, what should be the neuron type? "
                                    "(Your choices are {}): ".format(self.neuron_types))
                print(f"You picked {neuron_type}")
                
                self.cutoffs.append((characteristic, operator, comparison_val, neuron_type))