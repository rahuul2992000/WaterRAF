# WaterRAF
Retrieval-Augmented Water Level Forecasting for Everglades

Here's an overview of our Retrieval-Augmented Forecasting (RAF) framework for water level forecasting. 
![WaterRAF Framework](figure/framework.png)

## Installation

1. Create & activate your Python environment:
    - `conda create -n ENV_NAME python=3.10`
    - `conda activate ENV_NAME`

2. Install dependencies:
    - `pip install -r requirements.txt`

3. Verify your setup:
    - `python --version`  # should show Python 3.10.x
    - `pip list`          # confirm key packages are installed
   
## Directory Tree
```text
WaterRAF/                       - this repository root
├── README.md                   
├── requirements.txt            - pinned Python dependencies
├── Data/                       - raw & processed time-series data
│   └── final_2025_data.csv
├── ChronosBolt/                - adapter + exploration notebooks
│   ├── chronos_bolt_adapter.py
│   ├── Strategy-A-ChronosBolt.ipynb
│   ├── Strategy-B-ChronosBolt.ipynb
│   └── Strategy-C-ChronosBolt.ipynb
├── Visualizations/             - analysis & plotting notebooks
│   ├── Everglades-CorrelationPlots-Station-Variables-Analysis.ipynb
│   ├── RetDatabaseStudy-Heatmap.ipynb
│   ├── Visualization-Figure5-7days.ipynb
│   ├── Visualization-Figure5-14days.ipynb
│   ├── Visualization-Figure5-21days.ipynb
│   └── Visualization-Figure5-28days.ipynb
├── figure/                    - static framework image for paper/README
    └── framework.png
