This is a nascent application for analyzing electrophysiology experiments in Python.  Right now it's far from 
fully generalized and instead works with Itamar's Safety Recall PL cortex firing rate and LFP data, but hopefully in 
the future it can be abstracted to work with other experiments with between animal conditions, within animal 
conditions, and periods/blocks, events, and different types of neurons.

There are two basic categories of things this program can do: make graphs/figures and make csv files for further data  
analysis. The file that you execute to make one of those things happen is [main.py](main.py).  `main.py` imports all the 
functions from [procs](procs.py).  These functions in general assign opts, that is, dictionaries of options for building up data 
or graph display, then initialize a `Plotter` (or one of its descendants) to make one or more plots or an instance of 
`Stats` to make one or more csv files.  

The first step toward running a procs is to make sure you have set up your data.  This is currently happening in 
[initialize_experiment.py](initialize_experiment.py). The only rule for the order of data initialization is that an 
Animal is initialized with the obligatory arguments `name` and `condition` and the optional arguments 
`tone_period_onsets` and `tone_onsets_expanded`. After that, in more flexible order, a Group can be initialized with 

Before running a proc from main, ensure that you have correctly set all the opts.  Procs import their opts dicts from 
[opts_library](opts_library.py) by default, but you can change the opts in `opts_library` or override that behavior 
if you prefer them to come from elsewhere.  [opts.md](opts.md), currently in progress, explains the different opts 
available to you, and which are optional and which are obligatory.



***Notes for developers and troubleshooters***

The basic data representations can be found in three files, [spike.py](spike.py), which contains a hierarchical representation 
spike data with classes such as Animal, Unit, Period, and Trial, [lfp.py](lfp.py), which contains a hierarchical representation of
lfp data, with classes such as LFPAnimal, LFPPeriod, and MRLCalculator, and behavior, which contains a hierarchical representation
of behavioral data, with classes such as BehaviorPeriod.  Maybe someday this will change, but right now it's obligatory 
to define animals with the classes in spike.py before doing any other analysis, however it's not actually obligatory to 
define any spike data. Objects in these modules have parents and children.  For example, in the spike module, a Period's parent 
is an Animal, and its children are Trials.  Values for graphs are calculated on objects over their children; values for csv files

The Context class is defined in [context.py](context.py).  A context can keep track of state variables like `data_opts`, 
`neuron_type` and `period_type` and notify subscribers when they change. Each of the three types of data, spike, lfp, and
behavior, has a top level experiment class, whose responsibility it is to receive notifications of relevant context 
changes and update their descendants when necessary.  For example, when the current `neuron_type` changes to 'IN', the 
spike experiment will be notified and call a function on all the animals in the experiment to change the constituents of their `children` 
property to be just IN's.  This tends to me more important for making graphs and figures, which aggregate data by category, as 
opposed to csv files. `data_opts` also contains state variables that require updating; units update their constituent trials
depending on which trials are indicated for inclusion in `data_opts`.

Another module which builds on the content of `data_opts` is [data.py](data.py).  Most classes in the app inherit from 
`Base`, and nearly every class that is a data representation (e.g. `Period`, `LFPAnimal`) inherits from Data. (`TimeBin` 
is the sole exception).  `Base` contains property methods for accessing aspects of the data_opts or for setting/getting
aspects of the context.  For example, the `data_type` key in the `data_opts` dict is what informs the program what set 
of operations it's about to do. When `data_type` is set to 'power', for instance, the `data` property on any object will
return that objects `get_power` method (or raise an error if it does not exist).  The same for 'psth', 'mrl', etc. 
`Data` contains methods for establishing hierarchical relationships among data representations 
and for doing standard calculations (like calculating data means and standard deviations over an object's children.)

The modules containing the classes which coordinate the work to be done on the model are [plotters.py](plotters.py) and 
[stats.py](stats.py).  There are currently several types of platters in the `plotters` module (all of which inherit from
the top-level plotter class); so far `Stats` is just one class that handles both making csv files and interfacing with R for 
the purpose of running post-hoc tests.





