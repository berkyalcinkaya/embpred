# emb

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Introduction to Embryo Time Point Classification

Embryo time point classification from time-lapse imaging is a crucial aspect of assisted reproductive technology (ART) and in vitro fertilization (IVF) research. This project aims to classify different developmental stages of embryos using machine learning techniques applied to time-lapse images. Understanding these stages is essential for assessing embryo viability and potential chromosomal normality, which are critical factors in successful IVF outcomes.

## Embryo Developmental Stages

The project focuses on classifying the following key stages of embryo development:

1. **T1 (Single Cell)**: The initial stage after fertilization.
2. **tPN (Pronucleus Formation)**: Single cell with visible pronucleus formation.
3. **tPNF (Pronucleus Fading)**: Similar to tPN, but the pronucleus is starting to fade.
4. **T2 (Two Cell Stage)**: The embryo has divided into two cells.
5. **T3-T8 (Three to Eight Cells)**: Subsequent cell divisions, classified by the number of cells present.
6. **TB (Blastocyst Stage)**: Formation of the blastocyst.
7. **TEB (Expanded Blastocyst)**: Further expansion of the blastocyst.

## Importance of Timing in Embryo Development

Understanding the duration an embryo spends in each stage is crucial for several reasons:

1. **Predicting Embryo Viability**: Research has shown that the timing of key developmental events can be indicative of an embryo's viability. For instance, embryos that progress through these stages at an optimal rate are more likely to result in successful pregnancies.

2. **Assessing Chromosomal Normality**: The timing and pattern of cell divisions can provide insights into the chromosomal status of the embryo. Abnormal timing or irregular cell divisions may be associated with chromosomal abnormalities.

3. **Identifying Key Developmental Windows**: Certain stages, such as the transition from 4 to 8 cells, coincide with critical events like embryonic genome activation. The timing of these transitions can be indicative of the embryo's developmental potential.

4. **Refining Embryo Selection**: By analyzing the time spent in each stage, clinicians can potentially rank embryos within a cohort, allowing for more informed selection of embryos for transfer.

5. **Early Prediction of Blastocyst Formation**: The timing of early cleavage events (e.g., from T1 to T8) can be predictive of an embryo's likelihood to reach the blastocyst stage successfully.

## Machine Learning Approach

The project utilizes a dataset of 52 embryos from 41 patients, with images captured across 7 focal depths. Each 2D image is 800x800 pixels, with an average of 834 frames per embryo, spanning approximately 164 hours of development.

By training a machine learning model on this labeled dataset, the aim is to automatically classify each frame into one of the defined developmental stages. This automation can provide several benefits:

1. Consistency in classification across different clinics and embryologists.
2. Ability to process large amounts of data quickly and efficiently.
3. Potential to identify subtle patterns that may not be apparent to the human eye.
4. Possibility of developing predictive models for embryo viability based on developmental timing.

## Conclusion

Understanding and accurately classifying embryo developmental stages through machine learning has the potential to significantly improve IVF outcomes. By providing objective, quantitative assessments of embryo development, this technology could enhance the selection of the most viable embryos for transfer, potentially increasing pregnancy rates and reducing the risk of chromosomal abnormalities. As research in this field progresses, it may lead to more personalized and effective fertility treatments.


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for embpred
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── embpred                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes embpred a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

--------

