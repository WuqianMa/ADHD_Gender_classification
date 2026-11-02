"""Quantitative EDA adapted from notebook cells 61–112. No model training."""

import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_ind

from .common import TARGET_MAP


def run(df_q, df_s, dictionary, report):
    display = report.display

    Markdown = str

    # Notebook cell 61
    data_q = pd.merge(df_q, df_s, on='participant_id', how='inner')

    # Notebook cell 62
    display(data_q.shape)

    # Notebook cell 63
    display(data_q.columns)

    # Notebook cell 64
    display(data_q.describe())

    display('#### Observe:')

    display('## Missing values observation')

    # Notebook cell 67
    df_af = data_q[np.logical_and(data_q['ADHD_Outcome'] == 1, data_q['Sex_F'] == 1)]
    df_am = data_q[np.logical_and(data_q['ADHD_Outcome'] == 1, data_q['Sex_F'] == 0)]
    df_nf = data_q[np.logical_and(data_q['ADHD_Outcome'] == 0, data_q['Sex_F'] == 1)]
    df_nm = data_q[np.logical_and(data_q['ADHD_Outcome'] == 0, data_q['Sex_F'] == 0)]

    # Notebook cell 68
    dict_df_q = {'total': data_q, 'adhd-female': df_af, 'adhd-male': df_am, 'none-female': df_nf, 'none-male': df_nm}

    # Notebook cell 69
    def check_missing_values(dict):
        """
        This function checks and displays the missing values information for each subgroup in the input dictionary of dataframes.

        For each dataframe in the dictionary, the following information is displayed:
        1. The shape of the dataframe (number of rows and columns).
        2. The count of missing values for each column.
        3. The total number of rows in the dataframe.
        4. The number of rows with at least one missing value.
        5. The percentage of rows that would be dropped if rows with any missing values are removed.

        Parameters:
        dict (dict): A dictionary where keys represent subgroup names and values are pandas DataFrames.

        Returns:
        None: This function displays the information using the `display()` function but does not return any value.
        """
        for (key, df) in dict.items():
            display(Markdown(f'#### This is the infos of the **{key}** subgroup:'))
            display(Markdown('The shape is : \n'))
            display(df.shape)
            display(Markdown(f'Missing values count'))
            display(df.isnull().sum())
            rows_missing = df[df.isnull().any(axis=1)].shape[0]
            display(Markdown(f'Number of rows: \n **{df.shape[0]}**'))
            display(Markdown(f'Number of rows with missing values:  **{rows_missing}**'))
            drop_percentage = rows_missing / max(1, df.shape[0]) * 100
            display(Markdown(f'Percentage of rows to be imputed or dropped: **{drop_percentage:.2f}%**'))
            display(Markdown('___'))

    # Notebook cell 70
    display(Markdown('### If keep all the `MRI_Track_Age_at_Scan` feature'))
    check_missing_values(dict_df_q)

    # Notebook cell 71
    data_q_nm = data_q.copy().drop(columns=['MRI_Track_Age_at_Scan'])
    df_af_nm = df_af.copy().drop(columns=['MRI_Track_Age_at_Scan'])
    df_am_nm = df_am.copy().drop(columns=['MRI_Track_Age_at_Scan'])
    df_nf_nm = df_nf.copy().drop(columns=['MRI_Track_Age_at_Scan'])
    df_nm_nm = df_nm.copy().drop(columns=['MRI_Track_Age_at_Scan'])

    # Notebook cell 72
    dict_df_q_nm = {'total': data_q_nm, 'adhd-female': df_af_nm, 'adhd-male': df_am_nm, 'none-female': df_nf_nm, 'none-male': df_nm_nm}

    # Notebook cell 73
    display(Markdown('### If we drop `MRI_Track_Age_at_Scan`'))
    check_missing_values(dict_df_q_nm)

    display('#### To keep or to leave? ')

    # Notebook cell 75
    dic_q = dictionary[dictionary['DataType'] == 'Quantitative'][['Instrument', 'Field', 'Description']]
    display(dic_q)

    display('#### Attention: `MRI_Track,Age_at_Scan` should be replaced as `MRI_Track_Age_at_Scan`')

    # Notebook cell 77
    dic_q = dic_q.copy()
    dic_q['Field'] = dic_q['Field'].replace({'MRI_Track,Age_at_Scan': 'MRI_Track_Age_at_Scan'})

    display('#### Since this features stands for the age at time of MRI scan, we need to consider if **age** is a key factor in this study.')

    display('## Violin Plots for Each Feature distribution')

    # Notebook cell 80
    desc_dict = dic_q.set_index('Field')['Description'].to_dict()

    # Notebook cell 81
    def plot_split_violin_by_groups(df, x_col, hue_col, label_map=TARGET_MAP, exclude_cols=None, desc_dict=desc_dict):
        """
        Plots violin plots where:
        - x_col: categorical axis (e.g. ADHD_Outcome or Sex_F)
        - hue_col: variable to split violins
        - label_map: maps 0/1 to human labels
        - exclude_cols: additional fields to skip
        - desc_dict: maps field names to human-readable descriptions
        """
        df = df.copy()
        if label_map:
            for (col, mapping) in label_map.items():
                df[col + '_label'] = df[col].map(mapping)
        else:
            df[x_col + '_label'] = df[x_col]
            df[hue_col + '_label'] = df[hue_col]
        x_label_col = x_col + '_label'
        hue_label_col = hue_col + '_label'
        default_exclude = ['participant_id', x_col, hue_col]
        if exclude_cols:
            excluded = list(set(default_exclude + exclude_cols))
        else:
            excluded = default_exclude
        features = df.select_dtypes(include='number').columns
        features = [col for col in features if col not in excluded]
        for feature in features:
            subset = df[[feature, x_label_col, hue_label_col]].dropna()
            if subset.empty or subset[hue_label_col].nunique() < 2:
                continue
            description = desc_dict.get(feature, feature)
            display(Markdown(f'#### `{feature}` — *{description}*'))
            display(Markdown(f'#### Violin Plot for `{feature}` ({x_label_col} × split by {hue_label_col})'))
            plt.figure(figsize=(8, 6))
            sns.violinplot(x=x_label_col, y=feature, hue=hue_label_col, data=subset, split=True, palette='Set2')
            plt.xlabel(x_label_col.replace('_label', '').replace('_', ' ').title())
            plt.ylabel(description)
            plt.legend(title=hue_label_col.replace('_label', '').replace('_', ' ').title())
            plt.tight_layout()
            report.figure()
            display(Markdown('___'))

    display('### ADHD VS NON-ADHD (split by gender)')

    # Notebook cell 83
    plot_split_violin_by_groups(df=data_q, x_col='ADHD_Outcome', hue_col='Sex_F')

    display('### Female vs male ( split by ADHD vs non ADHD)')

    # Notebook cell 85
    plot_split_violin_by_groups(df=data_q, x_col='Sex_F', hue_col='ADHD_Outcome')

    display('### Summary')

    display('### Strong ADHD-related Features')

    display("#### Also, It seems like this dataset isn't strong at gender prediction, from visual plots.")

    display('## **Correclation check for qualidative dataset**')

    display('### r (correlation coefficient)')

    display('### Pearson (Linear) vs. Spearman (Monotonic)')

    # Notebook cell 92
    features = [col for col in data_q.columns if col not in ['participant_id', 'Sex_F', 'ADHD_Outcome']]

    display('## **Correlation: Features vs Targets**')

    # Notebook cell 95
    result_spearmanr = []
    result_pearsonr = []
    for feature in features:
        for target in ['Sex_F', 'ADHD_Outcome']:
            df_tmp = data_q[[feature, target]].dropna()
            if len(df_tmp) > 1:
                (r_pearson, pval_pearson) = pearsonr(df_tmp[feature], df_tmp[target])
                (r_spearman, pval_spearman) = spearmanr(df_tmp[feature], df_tmp[target])
            else:
                (r_pearson, pval_pearson) = (float('nan'), float('nan'))
                (r_spearman, pval_spearman) = (float('nan'), float('nan'))
            result_pearsonr.append({'feature': feature, 'target': target, 'pearsonr_r': r_pearson, 'p_value': pval_pearson})
            result_spearmanr.append({'feature': feature, 'target': target, 'spearmanr_r': r_spearman, 'p_value': pval_spearman})

    # Notebook cell 96
    def generate_correlation_dataframe(result_pearsonr, result_spearmanr, alpha=0.05):
        """
        Generate a DataFrame comparing Pearson and Spearman correlation results.

        Args:
            result_pearsonr (list): list of dicts with keys ['feature', 'target', 'pearsonr_r', 'p_value']
            result_spearmanr (list): list of dicts with keys ['feature', 'target', 'spearmanr_r', 'p_value']
            alpha (float): significance level, default 0.05

        Returns:
            pd.DataFrame: Correlation comparison table with strength.
        """
        import pandas as pd
        df_pearson = pd.DataFrame(result_pearsonr)
        df_spearman = pd.DataFrame(result_spearmanr)
        df = pd.merge(df_pearson, df_spearman, on=['feature', 'target'], suffixes=('_pearson', '_spearman'))
        correlation_type = []
        strength = []
        for (_, row) in df.iterrows():
            if pd.notna(row['pearsonr_r']) and row['p_value_pearson'] < alpha:
                correlation_type.append('🟢 Linear (Pearson)')
                r_used = abs(row['pearsonr_r'])
            elif pd.notna(row['spearmanr_r']) and row['p_value_spearman'] < alpha:
                correlation_type.append('🟡 Non-linear (Spearman)')
                r_used = abs(row['spearmanr_r'])
            else:
                correlation_type.append('🔴 No significant correlation')
                r_used = None
            if r_used is None:
                strength.append('No correlation')
            elif r_used >= 0.5:
                strength.append('Strong')
            elif r_used >= 0.3:
                strength.append('Medium')
            elif r_used >= 0.1:
                strength.append('Weak')
            else:
                strength.append('Very weak')
        df['correlation_type'] = correlation_type
        df['strength'] = strength
        return df

    # Notebook cell 97
    def display_correlation_by_target(df_correlation):
        """
        Display the correlation dataframe grouped by target, showing feature and correlation type.

        Args:
            df_correlation (pd.DataFrame): The correlation result dataframe.
        """
        targets = df_correlation['target'].unique()
        for target in targets:
            display(Markdown(f'\n### Target: `{target}`\n'))
            df_target = df_correlation[df_correlation['target'] == target]
            for (_, row) in df_target.iterrows():
                feature = row['feature']
                correlation_type = row['correlation_type']
                display(Markdown(f'- Feature: `{feature}` → {correlation_type}'))

    # Notebook cell 98
    def summarize_correlation_by_target_markdown(df_correlation):
        """
        Summarize features by target and correlation type (linear, non-linear, no significant correlation),
        and display nicely in Markdown format, including strength.
        """
        targets = df_correlation['target'].unique()
        for target in targets:
            md_text = f'### Target: `{target}`\n\n'
            df_target = df_correlation[df_correlation['target'] == target]
            df_linear = df_target[df_target['correlation_type'].str.contains('Linear')]
            md_text += '### 🟢 Linear Correlations (Pearson significant):\n\n'
            if not df_linear.empty:
                for (_, row) in df_linear.iterrows():
                    feature = row['feature']
                    strength = row['strength']
                    md_text += f'- `{feature}` (**{strength}**)\n'
            else:
                md_text += '_No linear correlations found._\n'
            df_nonlinear = df_target[df_target['correlation_type'].str.contains('Non-linear')]
            md_text += '\n\n### 🟡 Non-linear Correlations (Spearman significant only):\n\n'
            if not df_nonlinear.empty:
                for (_, row) in df_nonlinear.iterrows():
                    feature = row['feature']
                    strength = row['strength']
                    md_text += f'- `{feature}` (**{strength}**)\n'
            else:
                md_text += '_No non-linear correlations found._\n'
            df_nosig = df_target[df_target['correlation_type'].str.contains('No significant')]
            md_text += '\n\n### 🔴 No Significant Correlation:\n\n'
            if not df_nosig.empty:
                for (_, row) in df_nosig.iterrows():
                    feature = row['feature']
                    strength = row['strength']
                    md_text += f'- `{feature}` (**{strength}**)\n'
            else:
                md_text += '_All features have significant correlations._\n'
            display(Markdown(md_text))

    # Notebook cell 99
    def get_significant_linear_features_by_target(df_correlation):
        """
        Get lists of features that are significantly linearly correlated with each target,
        grouped by strength.

        Args:
            df_correlation (pd.DataFrame): Correlation dataframe.

        Returns:
            dict: Nested dictionary {target: {strength: [list_of_features]}}
        """
        significant_features = {}
        targets = df_correlation['target'].unique()
        strength_levels = ['Strong', 'Medium', 'Weak', 'Very weak']
        for target in targets:
            df_target = df_correlation[df_correlation['target'] == target]
            df_linear = df_target[df_target['correlation_type'].str.contains('Linear')]
            strength_dict = {level: [] for level in strength_levels}
            for (_, row) in df_linear.iterrows():
                feature = row['feature']
                strength = row['strength']
                if strength in strength_levels:
                    strength_dict[strength].append(feature)
            significant_features[target] = strength_dict
        return significant_features

    # Notebook cell 100
    df_correlation = generate_correlation_dataframe(result_pearsonr, result_spearmanr)

    # Notebook cell 101
    display(df_correlation)

    # Notebook cell 102
    significant_features = get_significant_linear_features_by_target(df_correlation)
    sex_features = significant_features['Sex_F']
    adhd_features = significant_features['ADHD_Outcome']

    # Notebook cell 103
    display_correlation_by_target(df_correlation)

    # Notebook cell 104
    summarize_correlation_by_target_markdown(df_correlation)

    display('## **Correlation: Target vs Target**')

    # Notebook cell 107
    df_target = data_q[['ADHD_Outcome', 'Sex_F']]
    (corr_coeff, p_value) = pearsonr(df_target['ADHD_Outcome'], df_target['Sex_F'])
    display(Markdown(f'#### The correlation is: \n  **{corr_coeff}** \n #### The confidence level is \n **{p_value}**'))

    display('### Summary:')

    display('## **Correltaion: Feature vs Feature**')

    # Notebook cell 111
    df_features = data_q[features].copy()
    correlation_matrix = df_features.corr(method='pearson')

    # Notebook cell 112
    plt.figure(figsize=(20, 18))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, linecolor='lightgrey', cbar=True)
    plt.title('Feature Correlation Matrix', fontsize=30)
    plt.xticks(rotation=60, ha='right', fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    report.figure()
