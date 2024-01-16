# Configuration for all analyses #


## Plots

If you are plotting, your config must be a dictionary.  It must include the key "data_opts", which must have as its 
value the data_opts dictionary.  It must also include the key "graph_opts". The data_opts dictionary must include the 
key "data_type" -- for instance, "psth", "mrl", etc.  This is the type of calculation you are doing.  For right now it must also include 
the key "data_class" ("spike", "lfp", or "behavior").

## Spreadsheets

The config can be either a dictionary of the form `{"data_opts": {"data_type": "<data_type>", ...}}` or a list of 
data_opts dictionaries (`[{"data_type": "<data_type>", ...}, {"data_type": "<data_type>", ...}...]`). If you are 
including multiple kinds of calculations the simplest way to handle them is to make the config a list of dictionaries 
that each have the config for an individual analysis, but there are other options.  As described above, these 
"data_opts" dictionaries must have the data_class and data_type specified.

## Multiple analyses

To learn more about configuration for analyses that need to iterate over multiple versions of `data_opts`, click here.

# Spike Analyses #

## PSTH ##

Here is a sample configuration file that could be used either to plot the PSTH at the group level, or generate a 
spreadsheet with PSTH values broken out by event and time bin.  

```
{
  "data_opts":  {"data_class": "spike", "data_type": "psth", "bin_size": 0.01, "adjustment": "normalized", 
  "events": {"pretone": {"pre_stim": 0, "post_stim": 1}, "tone": {"pre_stim": .05, "post_stim": .65}}, 
  "time_type": "continuous", "row_type": "event", "levels": ["group"], "block_types": {"tone": [0,1,2,3,4]}, 
  "data_path": "</path/where/csvs/are/written>}, 
  "graph_opts": {"graph_dir": "/Users/katie/likhtik/data/graphs", "units_in_fig": 4, "tick_step": 0.1, "sem": false, 
  "footer": true, "equal_y_scales": true, "group_colors": {"control": "#76BD4E", "stressed": "#F2A354"}}
}
```

### Opts for both plotting and making spreadsheets ###

"bin_size": the size in seconds of the histogram bins into which to place spike events for further analyses.

"events" (optional): a dictionary to specify the event structure for each period type, with period types as keys and 
nested dictionaries as values.  The dictionaries contain three values that are strictly optional but two of which you 
probably want to specify.  These two are "pre_stim" and "post_stim": the time in seconds before and after the onset of 
an event you want to take data from.  A pre_stim of .05 will take 50 ms before event start.  A post-stim of .65 will 
take 650 ms after the event.  A pre_stim of -.05 would start 50 ms after the event. The current defaults are 0 and 1 for 
pre and post, respectively. If you would like to take data from your entire reference period, but only data from the 
period surrounding the stimulus in your target period, you can specify the event characteristics as in the above example:
the pretone data will be the entire period, without interruption, but the tone data will only be the .05s before and the 
.65s after the stimulus.

The last key is "selection" -- if input is in JSON, an array enclosed in brackets, or if in Python, some kind of 
iterable with arguments for Python's [`slice` function](https://www.w3schools.com/python/ref_func_slice.asp), that will 
indicate which events for the relevant period_type to use in the analysis.  For example, if there are 5 tone periods 30 
events each, but you are only interested in the first period, you can specify "selection": [0, 30]. The default is to 
take all events. A sample key, value pair could look like 
```
"events": {"pretone": {"pre_stim": .05, "post_stim": .65, "selection": [0, 150]},
           "tone": {"pre_stim": .05, "post_stim": .65, "selection": [0, 150]}}
```

"adjustment" (optional): You can extract 
- raw rates ("adjustment" : "none"), 
- rates of experimental periods with the rates of the reference period subtracted, for example, tone periods with pretone periods subtracted ("adjustment":"relative"), 
or 
- normalized rates, which are further divided by the standard deviation of the firing rate over all experimental time 
from periods of the same type (in this example, the standard deviation of firing in all of the tone period experimental 
time.) The default is "normalized".

"spontaneous" (optional): an integer or an array/Python iterable that indicates that rather than analyzing time after 
the beginning of the presentation of stimuli, you are analyzing spontaneous activity. If it is an integer, it will take 
that many seconds before the first period in the experiment (that is not a reference period). If it is an iterable of 
length two, it will define the beginning and end, in seconds, of the period you want to analyze.  

### Opts for the plotting function, `plot_psth` ###

"levels" (obligatory for plotting): the levels of the data hierarchy at which you want to generate plots -- a JSON array
or, if passing opts in Python, any kind of iterable.  These can take the values "group," "animal," and "unit."  
There is thus far no option to plot periods or events separately, but unit plots include a spike raster that show every 
event over the selected period type.  Alternatively, if you only want to plot one level, you may specify "level" rather 
than "levels". '"levels":["group"]' and '"level":"group"' are equivalent.

"periods" (optional): a dictionary whose keys are the period types which will be included in the analysis for plots, and
whose values are arrays/Python iterables with the integer indices of the periods to include. Ignored for making CSVs. 
Although not strictly obligatory for plotting, if not provided, values from all periods will be included, which would be 
particularly nonsensical if any value for "adjustment" other than "none" is chosen. **Watch out for this. If you don't 
include it, the program will run without error, and you will get results that make sense, but they will be wrong.**


### Opts for `make_spreadsheet` ###

"row_type" (obligatory for spreadsheet construction): the level of the data hierarchy a row in your csv file represents. 
If you specify "unit", for instance, you would get values for units that had been averaged over 
events and periods.  This can take the values "group," "animal," "unit," "period", and "event," or in the case of cross-
correlations, "group," "animal," and "unit_pair".

"time_type" (optional): one of "block" or "continuous". It determines whether to further divide your data into time 
bins around the stimulus.  This option is ignored for the plotting function, which, true to the name, always plots data 
as a histogram around the stimulus, but for csv files, the default is average over time bins, and if you want to further
break out your data, "continuous" must be specified. 

"data_path" (optional): the path where the csv file will be written.  This is optional because you can pass it as an 
argument to Runner.run(), but it must be somewhere.


## PROPORTION ##

Configuration is as for PSTH with an addition and a change.  

"base" (optional): the level of the hierarchy whose proportion of being upregulated is being calculated.  Most commonly 
"event" or "unit".  Defaults to event, so that proportion is the proportion of upregulated events.

Note: it would make sense to make the definition of up- or down-regulated configurable, in which case that would become 
an opt here.  Right now upregulated means greater than 0. Also, be sure to note that the meaning of > 0 changes 
depending on "adjustment".

"evoked" (optional): a boolean which indicates whether to subtract the values from the reference period. **It does not make sense 
to use this if you have chosen something other than 'none' for adjustment**. Keep in mind: "adjustment" applies to the 
*underlying rates*, while the "evoked" subtraction will be applied to the further calculation you do using those rates 
(this is also relevant for the various kind of correlation calculations described below). The default is False.

Here's an example proportion config:

```{
  "data_opts":  {"data_class": "spike", "data_type": "proportion", "bin_size": 0.01, "adjustment": "normalized",
   "events": {"pretone": {"pre_stim": 0, "post_stim": 1}, "tone": {"pre_stim": .05, "post_stim": .65}},
    "time_type": "continuous", "row_type": "event", "levels": ["group"], "block_types": ["tone"]}
```

## AUTOCORRELATION ##

Configuration is as for PSTH with two additions.

"max_lag": an integer showing the number of lags to display.  If, for instance, stimuli (represented by events) are 1 
second apart and you choose 10 ms time bins, 99 is a sensible `max_lag`, because at 100 lags autocorrelation will likely 
spike up in response to the stimulus.

"base" (optional): as with proportion, you can choose at which level you want to perform the base calculation.  For 
instance, maybe you are interested in the autocorrelation of a unit's average rates, rather than the autocorrelation of 
rates during an individual event. You would then specify "base": "unit".  Default is events, which calculates the 
autocorrelation of the original time series of binned rates. 

A note about events specification: if you are interested in calculating autocorrelation using data from  a longer span 
of time than, for instance, a one-second stimulus duration, one good way to accomplish this is by setting pre_stim equal 
to 0, post_stim equal to your period duration, and make sure to select only one event per period with the 
"events" opt. Here is an example autocorrelation config:

```
{"data_class": "spike", "data_type": "autocorrelation",  'bin_size': 0.01, 'max_lag': 99,
"events": {"pretone": {"pre_stim": 0, "post_stim": 1, "selected": [0, 150, 30]}, 
"tone": {"pre_stim": .05, "post_stim": .65}, "selected": [0, 150, 30]},  'block_types': ['tone']}
```

"evoked": as described in the discussion of ["proportion"](#proportion)

## AUTOCORRELOGRAM ##

The same as for autocorrelation (just change "data_type" to "autocorrelogram").  

## SPECTRUM ##

"spectrum_series" (optional): the name of the time series to take a spectrum of -- default is "get_autocorrelation". All 
necessary opts for the underlying calculation must be specified. 'get_psth' would take a spectrum of the PSTH.

"freq_range": an array, or a Python iterable, indicating the endpoints of the range of frequencies through which to 
display the spectrum. It's a good idea to start above the frequency of the stimulus or it dwarfs other variation.  If 
you pick an upper frequency bound that's greater than the Nyquist frequency determined by the bin size, it won't cause
an error; the higher frequencies just won't be displayed.

"spectrum_base" (optional): a string that indicates which level of the hierarchy at which to stop averaging and take the
spectrum of the data for that object. The default behavior is to return a spectrum for the averaged data of the current 
object, so, for instance, if you are plotting the spectrum of group data you will get a spectrum of that group's average 
data.  If you would rather average over spectra, specify a "spectrum_base".  If you plotted group data and your 
"spectrum_base" were "unit", you would take the spectrum of the unit average for that data, and then average over units 
and animals to get a group value.

## CROSS-CORRELATION ##

Cross-correlations are performed with NumPy mode equal to 'full'. Configuration is as for autocorrelation, with one 
addition and one change.

"max_lag": the time in seconds to display on the cross-correlation graph (for autocorrelation it was an integer, the 
number of lags -- yes it would be great to make this consistent).

"unit_pair" (or "unit_pairs"): a string (or an array/iterable of strings) of the form 'NT1,NT2' that indicates what 
kinds of units to cross-correlate.  If, for instance, you are interested in seeing the relationship between acetycholine 
and parvalbumin neurons, labeled ACH and PV in your experiment, "unit_pair" would be "ACH,PV" (with ACH as the fixed 
time series and PV as the 'sliding' one).  

In this example configuration, the cross-correlations between ACH and PV cells are computed both for individual unit 
pairs and averaged over animal.  A smaller bin size and a max lag of 50 ms are used to see finer grained temporal 
dynamics between neurons.  All period types are selected -- if you were interested in understanding how 
cross-correlation changed via experimental condition you would take a different approach.  Adjustment is 'none' because 
we are interested in the raw data.  Pre-stim is 0 and post-stim is 1 because our event length is one second -- in this 
analysis we are taking all the data.

```
CROSS_CORR_OPTS = {'data_class': 'spike', 'data_type': 'cross-correlation', 'pre_stim': 0, 'post_stim': 1,
                   'adjustment': 'none', 'bin_size': 0.001, 'levels': ['animal', 'unit_pair'], 
                   'period_types': ['pretone', 'tone'], 'unit_pairs': ['ACH,PV'], 'max_lag': .05}
                   
```

## CORRELOGRAM ##

Same as for cross-correlation, just change "data_type" to "correlogram".


### Opts for `plot_group_stats` ###

These plots are, frankly, very specific to Itamar's analysis and likely to be quite brittle

Opts are as for `plot_psth` with some additions:

"post_hoc_bin_size" (optional):  
"post_hoc_type"
"data_path"

""

# LFP ANALYSIS # 

## Power ##

A note on implementation here.  Matlab Engine for Python is something that exists, but I wasn't able to get it to work 
and for current purposes have an acceptable alternative using the subprocess module to call Matlab from the command line.
It may be worth solving this problem at some point.

"matlab_configuration": a dictionary with the keys 
- "matlab_path": the full path to your Matlab executable 
- "program_path": the path to the directory where the Matlab program you want to execute is, or a higher level directory,
so long as every subdirectory can be added to the Matlab path without interfering with your program
- "temp_file_path": a path to a directory where a temp subdirectory will be written that will allow the creation of 
files for the program execution

"frequency_band" (string )

"power_deviation" (optional): a boolean that indicates whether to include in the CSV an idiosyncratic calculation that 
records how far above or below the local moving average a time bin is. Check the `lfp` module for `get_power_deviation` 
to see the implementation details. 




If you would like to iterate through different values for a key (for instance iterating through brain regions, or levels
("group", "animal", etc.)), you include the plural version of the key -- "brain_regions" or "levels" with a list you 
want to iterate through as a value, e.g. "brain_regions": ["il", "bla"].  You can include as many such plural keys as 
you wish and every combination of the list members will be iterated through.

If you need to change certain values dependent on others (for example, if you are making a csv file with results from 
multiple kinds of analyses that support different levels of time granularity), you must define a rule.  Rules take this 
form: {}