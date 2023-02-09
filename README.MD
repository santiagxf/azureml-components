# Azure Machine Learning components repository

This repository contains un-official Azure Machine Learning components to use in AzureML pipelines. The components contain a variety of functionalities to improve the current capabilities.


## Components

The following components are available.

| Category             | Module name                    | Description                                                                                             |
|----------------------|--------------------------------|---------------------------------------------------------------------------------------------------------|
| Scoring & Evaluation | [Evaluate model](components/evaluate/component.yml) | Performs the model evaluation of a given model and optionally it is compared against another model.     |
| Scoring & Evaluation | [Promote evaluation results](components/promote_evaluation/component.yml) | Promotes a given evaluation results to the pipeline run. Optionally, it can also promote all the tracking of the nodes in the pipeline. |

## Installation

### Create the base environment

The components in this reposity runs on the environment `components-base-py38`. You can create the environment with the AzureML CLI:

```bash
az ml environment create -f environments/components-base-py38.yml
```

### Create the components

You can install this modules using the AzureML CLI or AzureML studio.

```bash
cd <component folder> 
az ml component create -f component.yml
```