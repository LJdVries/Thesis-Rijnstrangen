# Thesis-Rijnstrangen

MSc thesis "Retention of Rhine water in the Rijnstrangen to mitigate drougth" by Laura de Vries.
Graduation September 27th 2021.

What is in this folder and how can I use it?
1. Datasets:
* Climate_2050GL		Contains the precipitation and evaporation data of the projected climate 2050GL.
* Climate_2050WH		Contains the precipitaiton and evaporation data of the projected climate 2050WH.
* Climate_Historical_DeBilt	Contains the historical precipitaiton and evaporation data.
* Climate_Ref2014		Contains the precipitaiton and evaporation data of the reference climate.

* Vol&Area_vs_depth_CSV		Contains the relation between the water level, volume of the basin and wet area of the basin.

* Waterstand_Historical_Lobith.DagGem	Contains the historical Rhine water levels at Lobith.
* Waterstand_Ref2017			Contains the Rhine water levels at Lobith of the reference climate.
* Waterstand_Rust2050(GL)		Contains the Rhine water levels at Lobith of the projected climate 2050GL.
* Waterstand_Stoom2050(WH)		Contains the Rhine water levels at Lobith of the projected climate 2050WH.

2. Python scripts:
I have run the scripts in PyCharm.

* Climate CSV generator
This little script can convert the original xlsm files (downloaded from knmi.nl (historical data) and 
meteobase.nl (corrected series)) to csv files. When using the datasets in this folder, this script is not needed.
Using csv files in the furter use, instead of the original xlsm files, decreases calculation times drastically.

* Datasets comparison
This script can read the various datasets and plot them. Examples of types of plots that can be generated are: 
datasets together, datasets separately, parts of the datasets, averages of the datasets, duration lines of the datasets, etc...
This script contains three functions: one for the Rhine water levels, one for the precipitation and one for the evaporation.

* GW comparison sections&layers
This script contains the comparison of the groundwater calculation for the cases with 7 and 9 sections and 1 and 3 layers, 
as presented in Appendix B of the thesis.

* GW sensitivity
This script runs the plots for the sensitivity analysis of the groundwater model to its parameters, as presented in 
Appendix C of the thesis.
In this script the groundwater function is calculated 1000 times with the input parameters randomized as defined, after which 
for each run the output is plotted against the input parameters. 

* GW validation
In this script the figures used for the groundwater validation are generated (Figure 5.1 and Figure 5.2 in the thesis).

* Water balance
This is the main script used for the groundwater calculations. A sequence of functions is used for reading and processing 
the data. Commented text with two hastags (##) gives explainations on the possible input of the functions and the use of it.
Commented text is used as well to explain what steps are taken in the funcitons and what units the data has. 
In the groundwater function in this script (starting on line 156) the option to use a mutli-layer system is included with 
commented lines. 
The volume of the basin and the fluxes are calculated in the volume_Rijnstrangen function. The results of this function can 
be plotted with the plot_volume_Rijnstrangen function.
The parameters and variables are defined below the functions, after which the functions are used and executed. 
