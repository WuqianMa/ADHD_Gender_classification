# Competition context

**WiDS Datathon 2025**

*Unraveling the Mysteries of the Female Brain: Sex Patterns in ADHD*

The competition asks participants to predict two binary targets from the
Healthy Brain Network data supplied for the datathon.

| Target | 0 | 1 |
| --- | --- | --- |
| `ADHD_Outcome` | Other / no ADHD diagnosis | ADHD |
| `Sex_F` | Male | Female |

## Data sources

| Data | What it contains | Role in this project |
| --- | --- | --- |
| Categorical metadata | Demographics and family background | Describe the study sample in EDA |
| Quantitative metadata | Questionnaire scores, handedness, age and related measurements | ADHD model inputs and EDA |
| Functional connectome matrices | Correlations between functional MRI brain regions | Sex_F model inputs and network EDA |
| Training solutions | ADHD and sex labels, matched by participant ID | Supervised training and evaluation |
| Data dictionary | Field descriptions, instruments and coded-value labels | Interpret features and EDA tables |

The original description lists 1,200+ training participants and 300+ test
participants. Test labels are withheld for competition scoring. A sample
submission gives the required output format.

## Dataset release

Use the revised training data supplied on March 20, 2025 (`TRAIN_NEW` or
`train_new_tsv`). The loaders also support the corresponding `TEST` / `test_tsv`
files and prefer the revised training release when both releases are present.

## Credits

The WiDS Datathon was developed with the Ann S. Bowers Women's Brain Health
Initiative, Cornell University and UC Santa Barbara. Data and support were
provided by the Healthy Brain Network, a Child Mind Institute initiative,
and the Reproducible Brain Charts project.

[Back to the project README](../readme.md)
