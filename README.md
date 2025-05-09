# Evaluation Alignment for Regression

## Overview
`eval-alignment-regression` is a project focused on evaluating and analyzing alignment regression models. The project includes data preprocessing, model training, predictions, and visualization of results. It is designed to handle datasets related to tuna pricing and demand, with support for historical exchange rate adjustments and other features.


## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required Python libraries (listed in `requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/eval-alignment-regression.git
   cd eval-alignment-regression

2. Create a virtual environment and activate it:
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

Usage:

All alignment experiments can be found in `alignment_properscoring.py`. 

Before alignment script can be run the corresponding predictive modeling and optimization task for each experiment should be run from its own directory:
- `tuna data`: Inventory Optimization with real data
- `synthetic tuna data`: Inventory Optimization with synthetic data 
- `synthetic downstream`: Synthetic experiment based on twCRPS. 

`illustration.py` is the toy example on the first page of the paper.