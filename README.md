# Covid19 Models

This repository contains a collection of Python Notebooks and code to do visualization and modeling using publicly available data for the covid19 outbreak.
Click [here](./INSTALL.md) for instructions on installing the notebook environment and preprocessing data to do visualizations.

## SIR Model

![SIR Example](./images/sir_model.png)

An SIR model is a set of differential equations used to model the spread of disease through a population. The model keeps track of three different subsets of the population over time:

* (S)usceptible: People who have not yet been infected and are susceptible to the disease.
* (I)nfected: People who currently have the disease and are contagious.
* (R)ecovered: People who have recovered from the disease and are no longer contagious.

[This notebook](./sir_modeling.ipynb) attempts to model the progression of deaths from COVID-19 for different regions of interest, from county and state levels in the US, to countries around the world. In theory, an SIR model can be used to predict 4 different metrics when fitted to actual data, because the model is parameterized by 4 different values. However, keep in mind that models are rarely accurate until they are starting to head over the peak death rate, since many parameters will fit the exponential growth portion very well. Also, if a modeled region does not have enough deaths per day, or the day-to-day numbers have high variance, the model can also have many "local optimas" that make it hard to know what is the right set of parameters. Some higher level aggregation across multiple regions can probably help improve model stability and accuracy.

Italy fits well because they are far ahead along on the curve:
![Italy Total Deaths](./images/italy_deaths_total.png)
![Italy Death Rate](./images/italy_death_rate.png)

### The 4 parameters

1. **Initial susceptible population:** When modeling deaths, this is the population of individuals who will eventually die from COVID-19. Dividing this value by the total population of the region can give us the *population fatality rate*. If the same fatality rate per infected individual is assumed across all regions, then this can tell us how good this region is at *suppressing* the disease, i.e. preventing parts of their population from ever getting the disease before the virus starves to death.
2. **Average time of contagion:** This parameter determines how long an average infected person remains contagious. This metric should be the sum of the incubation period, symptomatic period, and contagious period after symptoms disappear.
3. **Average rate of spread:** This parameter can be thought of as the average number of people a person infects over a time period (e.g. 1 day). It can be influenced by population density, cultural aspects around distance and comfort zone, social distancing measures, wearing of masks or PPE, and other types of human social and physical behavior.
4. **Initial infected population:** The datasets start on the first date that a positive case has been reported, which we call time 0. The initial infected population, or the initial number of people who are infected and will eventually die at time 0, is another way to tell us how responsive the country's CDC is in catching and reporting cases. If the number is high, it means that potential many infections have already taken place before the country/region's CDC started reporting any cases.

## Acknowledgements

[COVID-19 Data Github](https://github.com/CSSEGISandData/COVID-19): This is the project's data source for COVID-19 cases and deaths, and is updated daily.

