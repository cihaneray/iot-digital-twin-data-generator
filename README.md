# Data Generation For IIoT Digital Twin

> Inspired by [Azure AI-PredictiveMaintenance](https://github.com/Azure/AI-PredictiveMaintenance/blob/master/src/Notebooks/DataGeneration.ipynb)

## Industrial machinery simulator for AI-enabled IoT scenarios

### Introduction

Implementing a production IoT Predictive Maintenance solution requires having mature remote condition monitoring
infrastructure in place. More specifically, data-driven prognostics hinge on the availability the provision from
statistically
significant amounts of run-to-failure telemetry and maintenance records, from which equipment degradation patterns can
be learned to enable failure predications based on both historical and new collected data.
Real-world run-to-failure data sets are virtually impossible to come across due to their commercially sensitive nature.
Of the several publicly available synthetic data sets, none ideally fits the canonical IoT scenario in which highly
irregular real-time data streams from sensors are captured and used for condition monitoring or
anomaly detection. For instance, the tiny *Turbofan Engine Degradation Simulation Data Set*
created with the commercial version of the *Modular Aero-Propulsion System Simulation* environment (C-MAPSS) contains a
single telemetry snapshot per operational cycle of unspecified length, whereas the
data set which follows the *Predictive Maintenance Modeling Guide* features hourly, presumably
aggregated, sensor readings produced by making statistical alterations to uncorrelated series of random numbers.
Unfortunately, the source code used for generating these data sets is not publicly available (in fact, getting C-MAPSS
requires a grant from the U.S. Government). This makes the tasks of converting these data sets to a format which bears
more resemblance to that of an IoT-centered scenario and generating new customized data for real-time simulation rather
difficult.
The approach to data generation presented in this Notebook is an attempt to simulate a hypothetical IoT-enabled
manufacturing environment comprising generalized rotational equipment, which may include pumps, turbines, gearboxes,
compressors, and engines. The end product is a customizable open-source physics-inspired simulation framework that can
be used for generating arbitrarily large seed data for model training as well as real-time telemetry streams for
performing interactive end-to-end testing and validation of a complete AI-enabled IoT solution. (Suitability for RUL,
multiclass classification and failure detection – elaborate.)

### Data inventory

The generated data set will be a combination of maintenance and telemetry records. It is assumed that sensor data, along
with the values describing a machine's current operational settings (in this case, rotational speed), is periodically
transmitted from an IoT Edge device, with or without preprocessing, to the cloud where it is used for predictive
analysis.

#### Maintenance and failure records

In the present implementation, the maintenance data set will contain primarily failure events indicating exact points in
time when a machine had a critical failure of a particular type. The intent of Predictive Maintenance is to prevent
these events by raising alarms in advance so that appropriate preventive activities can be carried out.

##### Format

- timestamp
- level (INFO, WARNING, ERROR, CRITICAL)
- machine ID
- code (identifies event/failure type)
- message (contains additional details)

#### Telemetry

This data set will represent an IoT telemetry stream.

##### Format

- timestamp
- machine ID
- ambient temperature (°C)
- ambient pressure (kPa)
- rotational speed (RPM)
- temperature (°C)
- pressure (kPa)
- vibration signal

### Simulation techniques overview (optional)

This section outlines the data generation methodology; it is included for completeness and is not required for
understanding the transformations presented in the Notebooks which reference the generated data sets. This topic may be
of interest to those who intend to customize, enhance or augment the data for a different scenario. (Skip to data
generation for everyone else.)

#### Time-varying failure propagation

For a physics-inspired data generation, similarly to the approach in *Damage Propagation Modeling for Aircraft Engine
Run-to-Failure Simulation*, a generalized equation for wear, $w = Ae^{B(t)}$, will be used to model
observable phenomena associated with equipment degradation over time.
This generalized wear equation ignores micro-level processes but retains macro-level degradation characteristics. It can
be rewritten as a time-varying health index
$$
h(t)=1 - d - exp\\{at^b\\}
$$
where

- $d$ is the initial degradation
- $a$ and $b$ are the coefficients defining the trajectory of health loss