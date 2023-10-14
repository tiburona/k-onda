## data opts


data_class: 'spike', 'behavior', or 'lfp'.  Obligatory.

data_type: a string indication the type of data to be analyzed.  Used to determine the method called by an object's 
`data` property.  Current options: 'psth', 'autocorr', 'spectrum', 'power', 'mrl', 'percent_freezing'. Obligatory.

data_path: The top-level path to which graphs and spreadsheets will be written (subdirectories are also created).  
Obligatory.  (Note: currently the Figure code doesn't write a figure to disk and data_path means something different 
in that context -- where to find the data for phy-related calculations.  This should be fixed.  #todo)

pre_stim: the amount of time (s) to take before the stimulus on a trial for a given data calculation. Obligatory.

post_stim: the amount of time (s) to take after the stimulus onset on a trial for a given data calculation. Obligatory.

bin_size: The length (s) of the histogram bins in for firing rate calculations. Obligatory for firing rate calculations, 
not for lfp.

trials: a two- or three-tuple that indicates which trials are selected for the current analysis. Unpacked, it should 
make sensible arguments to the Python `slice` function. Obligatory for spike calculations, selected only certain trials 
is not yet implemented for lfp power calculations.

adjustment: For spike data, the alteration to make to tone period data relative to pretone.  
'normalized', 'relative', or 'none'.  Optional. In the absence of a value, the program defaults to 'normalized'.

frequency_type: a parameter that, when specified as 'continuous' while generating a CSV file, will split a frequency
series into frequency bins.

time_type: a parameter that, when specified as 'continuous' while generating a CSV file, will split a time series into 
time bins 

selected_animals: a list of animals to include in the analysis.  When absent, all animals are included by default.

