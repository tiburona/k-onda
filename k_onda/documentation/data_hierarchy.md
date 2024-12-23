Experiment: a representation of the study you are running. Experiment and its descendants are also representations of
spike data.  Even if you're not analyzing spike data, the program initializes an experiment and its animals.

Animal: an animal in the experiment, which has associated spike data.

Unit: a cluster of spike data in the experiment.  The reason it's not called a cell or neuron here is that although no 
ability to analyze this data has been added to this application yet, technically units have the category attribute 'good'
or 'MUA' (as labeled by Phy) and some day MUA analysis functionality could be added.

Period: a period of time in the experiment. Period type, an attribute of period, is a within subject experimental 
condition.

Event: the time around the presentation of a stimulus in the experiment (or any other event with an onset at a 
time)

Time bin: a division of an event over which spikes are aggregated.  

Unit pair: two units whose cross correlation, or cross correlogram, can be calculated.

LFPExperiment: an analogue of experiment that represents LFP Data.

LFPAnimal: an analogue of Animal that represents LFPData.

LFPPeriod: an analogue of Period that represents LFPData.

MRLCalculator: an object that brings together the data from a spike Unit and an LFPPeriod to calculate the extent to 
which units are in phase with LFP signal.

LFPEvent: as you may have guessed, an analogue of LFPEvent.

Frequency bin: a slice of any higher level data representation by the smallest increment of frequency in the data (thus 
far 1 Hz)

Time bin: in the LFP context, a slice of any higher leve data representation by the smallest increment of time in the 
data (thus far .01 seconds).

BehaviorExperiment: an analogue of Experiment that represents behavior data

BehaviorAnimal: an analogue of Animal that represents behavior data

BehaviorPeriod: an analogue of Period that represents behavior data.  The application does not currently have the 
capacity to represent behavior at a finer resolution than Period.