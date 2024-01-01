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
  "data_opts":  {"data_class": "spike", "data_type": "psth", "pre_stim": 0.05, "post_stim": 0.65, "bin_size": 0.01,
    "tone_event_selection": [0, 300], "adjustment": "normalized", "time_type": "continuous", "row_type": "event",
    "levels": ["group"], "block_types": ["tone"], "data_path": "</path/where/csvs/are/written>},
  "graph_opts": {"graph_dir": "/Users/katie/likhtik/data/graphs", "units_in_fig": 4, "tick_step": 0.1, "sem": false,
    "footer": true, "equal_y_scales": true, "group_colors": {"control": "#76BD4E", "stressed": "#F2A354"}}
}
```

### Opts for both plotting and making spreadsheets ###

"pre_stim" and "post_stim": the time in seconds before and after the onset of an event you want to take data from.
A pre_stim of .05 will take 50 ms before event start.  A post-stim of .65 will take 650 ms after the event.  A pre_stim 
of -.05 would start 50 ms after the event.

"bin_size": the size in seconds of the histogram bins into which to place spike events for further analyses.

"<period_type>_event_selection" (optional): if input is in JSON, an array enclosed in brackets, or if in Python, some kind of 
iterable with arguments for Python's [`slice` function](https://www.w3schools.com/python/ref_func_slice.asp), that will 
indicate which events for the relevant period_type to use in the analysis.  For example, if there are 5 tone periods 
with 30 events each, but you are only interested in the first period, you can specify "tone_events": [0, 30].
The default is to take all events.  

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

"period_types" (optional): the period types which will be included in the analysis for plots. Ignored for making CSVs. 
Although not strictly obligatory for plotting, if not given values from all periods will be included, which would be 
particularly nonsensical if any value for "adjustment" other than "none" is chosen.


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

### Opts for `plot_group_stats` ###

Opts are as for `plot_psth` with some additions:

"post_hoc_bin_size": 
"post_hoc_type"
"data_path"

""


## PROPORTION ##

Configuration is as for PSTH with an addition.  

"base" (optional): the level of the hierarchy whose proportion of being upregulated is being calculated.  Most commonly 
"event" or "unit".  Defaults to event, so that proportion is the proportion of upregulated events.

Note: it would make sense to make the definition of up- or down-regulated configurable, in which case that would become 
an opt here.

```{
  "data_opts":  {"data_class": "spike", "data_type": "proportion", "pre_stim": 0.05, "post_stim": 0.65, "bin_size": 0.01,
    "tone_event_selection": [0, 300], "adjustment": "none", "time_type": "continuous", "row_type": "event",
    "levels": ["group"], "block_types": ["tone"]}
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
"<period_type>_events" opt.  See the `sample_autocorr_config` for an example.

## AUTOCORRELOGRAM ##

The same as for autocorrelation.  

## SPECTRUM ##

"spectrum_series" (optional): the name of the time series to take a spectrum of -- default is "get_autocorrelation". All 
necessary opts for the underlying calculation must be specified. 'get_psth' would take a spectrum of the PSTH.

"freq_range": an array, or a Python iterable, indicating the endpoints of the range of frequencies through which to 
display the spectrum. It's a good idea to start above the frequency of the stimulus or it dwarfs other variation.  If 
you pick an upper frequency bound that's greater than the Nyquist frequency determined by the bin size, it won't cause
an error; the higher frequencies just won't be displayed.  

## CROSS-CORRELATION ##

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