# Lab 10 — Data Preprocessing (CSV • Pandas)

This lab continues from Lab 9 and demonstrates **data preprocessing** for a CSV dataset using **Python + Pandas**. It focuses on:
- Handling **missing values** (nulls)
- Converting datatypes (especially converting columns to **integers**)
- Cleaning the dataset (dropping unnecessary columns)
- Preparing data by splitting into **X** (features) and **y** (target)
- Converting categorical/object columns into numeric integer codes

## Project Structure
```text
Lab-10-Data-Preprocessing/
├── data/
│   └── dataset.csv
├── notebooks/
│   └── lab10_data_preprocessing.ipynb
├── src/
│   └── preprocessing.py
├── outputs/
│   └── cleaned_dataset.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run
From the repository root:

```bash
python src/preprocessing.py
```

## What It Produces
After running, the script writes:
- `outputs/cleaned_dataset.csv`

And it prints to the console:
- rows/columns (shape)
- missing value counts
- unique values preview for selected columns
- datatypes before/after cleaning
- shapes of **X** and **y**

## Notes
- The script is written to be clean and GitHub-friendly.
- Missing values are filled using **mode** for categorical columns.
- Object columns are converted to integer codes using `pd.factorize()`.

