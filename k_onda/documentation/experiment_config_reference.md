# Experiment Configuration Reference 


Your experiment configuration is a dictionary, either JSON or a Python dictionary.

Note: this reference uses the world "iterable" as an umbrella term for what could be a json array defined in square 
brackets, or if you are defining your configuration in Python code, could be any kind of Python iterable, including but not 
limited to a list or a tuple. 

These are its keys.

## Top level keys

"conditions": an iterable with the names of the between groups conditions in your experiment, for example "control" and 
"treatment".

"identifier": just a name for your experiment

"neuron_types": if you're analyzing spike data, the putative kinds of neurons you're analyzing

"sampling_rate": the rate, in Hz, at which spike data was collected

"lfp_sampling_rate": the rate, in Hz, at which LFP data was collected

"lfp_root": the directory in which LFP data is stored.

"lfp_path_constructor": this program assumes that just below the lfp_root are subdirectories with the animal's 
identifier.  It further assumes that the LFP data we're trying to open is stored in BlackRock ns3 files. The LFP path 
constructor is an iterable that contains any subdirectories, and finally the name of the file, minus the extension. If 
the file name is the same as the animal identifier, you can put "identifier" and the program will replace it with the 
animal identifier.  So for example, if your data is stored like this: "/my_lfp_dir/recall_data/animal150.ns3", 
lfp_root would be "my_lfp_dir" and lfp_path_constructor would look like "["recall_data", "identifier"]".  If on the 
other hand all your animal files had the name "recall.ns3" then your lfp_path_constructor would look like 
"["recall_data", "recall"]".

"lost_signal": The amount of LFP signal, in seconds, that is lost at the edge of an mtcsg analysis, given your mtcsg 
arguments. This is going to need to be abstracted soon to allow the use of more than one set of arguments in the 
experiment.

"stimulus_duration": How long the stimulus lasts during an event.  If you have pips of .05 seconds every 1 second, 
for example, the stimulus is .05 seconds, and the event is 1 second.

"frequency_bands": a dictionary with the names of frequency bands as keys and their ranges as python iterables.  For 
example `{"delta":(0, 4), "theta_1":(4, 8)}`.

"behavior_data": a path to a CSV file that contains the behavior data.  Right now it only handles one value per period 
(see [here](data_hierarchy.md)) for a definition of a period.  One column must contain the identifier of each animal, 
and then period data should be stored in columns named <period name> <period number>, in ascending order of period.  
For example, Pretone 1, Pretone 2, Tone 1, Tone 2 or Pretone 1, Tone 1, Pretone 2, Tone 2 are both fine.  The presence 
of other columns will not cause an error.  The names are not case-sensitive.  No provisions have yet been made for
missing data.

'behavior_animal_id_column': The name of the column in the just-mentioned spreadsheet where the animals' identifiers are
stored, e.g. 'animal_id'.

"animals": a dictionary, the contents of which are described below

## Animal keys

"identifier": the animal's string identifier

"period_info": itself a dictionary, the contents of which are described below

"units": itself a dictionary, the contents of which are described below

"lfp_electrodes": itself a dictionary.  The keys are brain regions, and the values are the number (0-indexed) of the 
electrode that recorded that brain region.  For example, if the electrodes that recorded from BLA and HPC were 1 and 3,
respectively, this would read ```{'bla': 0, 'hpc': 1}```

"lfp_from_stereotrodes": itself a dictionary.  Keys are "nsx_num", i.e., the kind of nsx file you're taking the 
stereotrode recordings from, probably 5 or 6, and "electrodes".  The value for the electrode key is an iterable with the
electrodes (0-indexed) that have the stereotrode data.  They will be averaged together for the final value.  So an 
example such dictionary is ```{"nsx_num": 6, "electrodes": [2, 4]}```

So an animal dictionary looks like ```{"identifier": <animal_identifier>, "period_info":<period_info_dictionary>, 
"units": <units_dictionary>, "lfp_electrodes": <lfp_electrode_dictionary>, "lfp_from_stereotrodes": 
<lfp_from_stereotrode_dictionary>}```

## Period info keys

Periods can be of two types, relative and non-relative.  Pretone periods are examples of periods that are relative to 
tone periods.  The necessary keys are different for each of them.  The keys in the period_info dictionary will be the 
names of period types, and the values will themselves be dictionaries 

Non_relative periods must have keys: 

"onsets": an iterable of times in samples, not seconds, when the periods of this type began.  The sampling rate for 
this time is whatever you put for "sampling_rate" at the top level of experiment config.

"events": an iterable iterables.  It has as many elements as there are periods of this period type, and each of those 
elements is an iterable of the start times of events, in samples (absolute start times, not start times as measured from 
the beginning of the period).

"event_duration": the duration of each event, in seconds.  Right now only one event type per period is supported.

"lfp_padding": an iterable of length two with the amount of time in seconds to add to each end of an LFP period such 
that no data is lost to mtcsg.

"reference_period_type": the period type to use as reference when calculating evoked data

Relative periods must have keys

"relative": True, i.e., a boolean indicating whether the period is relative

"target": The kind of period its position is defined relative to.  For a "pretone" period, this is "tone".

"duration": as above

"lfp_padding": as above

An example period_info dictionary looks like 

```

{"pretone": {'relative': True, "target": "tone", "shift": 30, "duration": 30, "lfp_padding": [1, 1]}, 
"tone" : {"onsets": [<5 different onsets>], "events": [[<30 event start times>], [<30 event start times>], 
[<30 event start times>], [<30 event start times>], [<30 event start times>]], "duration": 30, "event_duration": 1, 
lfp_padding": [1, 1], "reference_period_type": "pretone"}
}
```

## Units keys

The units dictionary can have three keys, "good", "mua", and "noise", but it's fine if it only has "good".  The values
are lists of dictionaries for the individual units.  They have keys

"spike_times": an iterable, in samples of the times firing was recorded

"electrodes": an iterable of most informative/active electrodes for that cluster (0-indexed). (Thus far this has been 
determined in the Likhtik lab by looking at each cluster and manually making a spreadsheet of the best electrodes.)

"neuron_type": a string with the neuron's assignment to a neuron type.  Thus far this has been determined by a k-means
clustering in a Matlab script.

"FWHM_microseconds": the full width half minimum (because deflections are negative in our recording) of the neuron in 
microseconds.  This is determined by the aforementioned Matlab script (and used to categorize neurons).

An example units dictionary could thus look like this: 

```
{"units": "good": [{"spike_times": [<an iterable of times in samples>, "electrodes": [2, 4], "neuron_type": "PN", 
"FWHM_microseconds": 630}, {<another unit>}, ...]}
```
    





