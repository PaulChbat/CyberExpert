# CyberExpert Log Analyzer & Mitigation Suggester

## Overview

This project, potentially developed as a Final Year Project (FYP), provides an application (`CyberExpert`) that leverages a fine-tuned language model to analyze security event logs. Its primary functions are:

1.  **Summarize Event Logs:** Condense potentially verbose log data into concise summaries.
2.  **Identify Potential Threats:** Analyze the summarized logs to highlight potential security issues.
3.  **Suggest Mitigation Actions:** Based on the identified threats, recommend steps to mitigate them.

The core of the application relies on a model fine-tuned specifically for this cybersecurity task using the provided dataset.

## Features

* Automated processing of event logs.
* Intelligent summarization using a fine-tuned model.
* Threat identification within log data.
* Actionable mitigation suggestions.

## Project Structure

```text
.
├── APP/                  # Contains the main application code (CyberExpert)
│   └── ...             # (Placeholder: specific app files like main.py, utils.py, model loading logic etc.)
├── DATA/                 # Contains dataset and preparation scripts
│   ├── Dataset_final.json # The final dataset (10k records) used for fine-tuning
│   └── data_prep.py      # Python script for cleaning/preparing raw data into Dataset_final.json format
├── FYP_Fine_Tuning.ipynb # Jupyter Notebook detailing the model fine-tuning process
└── README.md             # This file (you are here!)
