"""Categorical EDA adapted from notebook cells 23–56. No model training."""

import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_ind

from .common import TARGET_MAP
import plotly.express as px
import plotly.graph_objects as go


def run(df_c, df_s, dictionary, report):
    display = report.display

    Markdown = str

    # Notebook cell 23
    df_c.info()

    # Notebook cell 24
    display(df_c.dtypes)

    # Notebook cell 27
    display(dictionary.columns)

    # Notebook cell 28
    display(dictionary['DataType'].unique())

    # Notebook cell 29
    df_checkinfo = dictionary[dictionary['DataType'].astype(str).str.strip().isin(['Categorical', 'Target'])][['Field', 'Description', 'Labels']].copy().reset_index(drop=True)

    # Notebook cell 30
    def parse_to_dict(s):
        """
        Convert a string with 'key=value' pairs into a dictionary.
        Handles both newline-separated and comma-separated formats.

        Parameters
        ----------
        s : str or NaN
            A string containing multiple 'key=value' pairs, separated by either newlines ('\\n') or commas (',').
            Example: '1=Apple\\n2=Banana' or '0= Other/None, 1=ADHD'

        Returns
        -------
        dict
            A dictionary where keys are integers and values are corresponding cleaned strings.

        Example
        -------
        >>> parse_to_dict("1=Apple\\n2=Banana")
        {1: 'Apple', 2: 'Banana'}

        >>> parse_to_dict("0= Other/None, 1=ADHD")
        {0: 'Other/None', 1: 'ADHD'}
        """
        if pd.isna(s):
            return {}
        if '\n' in s:
            entries = s.split('\n')
        else:
            entries = s.split(',')
        result = {}
        for entry in entries:
            if '=' in entry:
                (key, val) = entry.split('=', 1)
                result[int(key.strip())] = val.strip()
        return result

    # Notebook cell 31
    df_checkinfo['Labels'] = df_checkinfo['Labels'].apply(parse_to_dict)

    # Notebook cell 32
    display(df_checkinfo.head(3))

    # Notebook cell 33
    data_c = pd.merge(df_c, df_s, on='participant_id', how='inner')

    # Notebook cell 34
    display(data_c.dtypes)

    # Notebook cell 35
    for col in data_c.columns:
        if col != 'participant_id':
            data_c[col] = data_c[col].apply(lambda x: int(x) if pd.notna(x) else pd.NA)

    # Notebook cell 36
    for (_, row) in df_checkinfo.iterrows():
        field = row['Field']
        labels_dict = row['Labels']
        if field in data_c.columns and isinstance(labels_dict, dict) and labels_dict:
            data_c[field] = data_c[field].map(labels_dict).astype('category')

    # Notebook cell 37
    display(data_c.head(2))

    # Notebook cell 38
    field_to_description = dict(zip(df_checkinfo['Field'], df_checkinfo['Description']))

    # Notebook cell 39


    display('## Table of Category Frequencies')

    # Notebook cell 41
    count_results = {}
    for col in data_c.columns:
        if col != 'participant_id':
            df_ = data_c[col].value_counts().reset_index(name='count').rename(columns={'index': col})
            count_results[col] = df_

    # Notebook cell 42
    for (key, df) in count_results.items():
        display(Markdown(f'### {key} is {field_to_description.get(key, key)}'))
        display(df)

    display('## Bar Charts of Categorical Distributions')

    # Notebook cell 45
    def wrap_labels_html(labels, width=60):
        """
        Wraps a list of labels into multiple lines using HTML <br> tags,
        so they display nicely in interactive Plotly plots (especially long category labels on axes).

        Parameters
        ----------
        labels : list or iterable
            A list (or other iterable) of labels to wrap. Labels are converted to strings if not already.

        width : int, default=60
            The maximum number of characters per line before inserting a line break.
            Adjust this based on your plot size and readability preference.

        Returns
        -------
        wrapped_labels : list of str
            A list of strings where long labels are split into lines using <br> tags.
            Suitable for use in Plotly axis labels to allow line breaks.

        Example
        -------
        >>> wrap_labels_html(["Long occupation, very long title", "Short label"], width=20)
        ['Long occupation,<br>very long title', 'Short label']
        """
        return ['<br>'.join(textwrap.wrap(str(label), width=width)) for label in labels]

    # Notebook cell 46
    for col in data_c.columns:
        if col != 'participant_id':
            counts = data_c[col].value_counts().sort_values()
            fig = px.bar(x=counts.values, y=wrap_labels_html(counts.index), orientation='h', title=f'The count of the {field_to_description.get(col, col)}', labels={'x': 'count', 'y': field_to_description.get(col, col)})
            fig.update_layout(height=600, yaxis=dict(tickfont=dict(size=10)), margin=dict(l=20, r=20, t=50, b=50))
            report.figure(fig)

    display('### Observation:')

    display('## Comparing Categories Across Sex and Diagnosis')

    # Notebook cell 49
    crosstab_results = {}
    for col in data_c.columns:
        if col not in ['participant_id', 'Sex_F', 'ADHD_Outcome']:
            cross_counts = pd.crosstab(index=data_c[col], columns=[data_c['Sex_F'], data_c['ADHD_Outcome']])
            cross_counts.index.name = col
            crosstab_results[col] = cross_counts

    # Notebook cell 50
    for (key, df) in crosstab_results.items():
        display(Markdown(f'### The {key} is the {field_to_description.get(key, key)}\n'))
        display(crosstab_results[key])

    # Notebook cell 51


    display('## Stacked Bar Plots: Gender and Diagnosis Breakdown')

    # Notebook cell 53
    def flatten_crosstab_simple(crosstab_df, index_col_name=None, sep='_'):
        """
        Flattens a crosstab DataFrame with MultiIndex columns into a long-form tidy DataFrame.

        This version flattens column MultiIndex into string labels like 'Female_ADHD',
        and optionally splits them back into group variables.

        Parameters
        ----------
        crosstab_df : pd.DataFrame
            The crosstab DataFrame to flatten. Must have a named index.

        index_col_name : str, optional
            The name of the index column. If None, will use crosstab_df.index.name.

        sep : str, default='_'
            Separator used to join MultiIndex column values.

        Returns
        -------
        pd.DataFrame
            A tidy DataFrame with columns: [index_col, Group, count, group1, group2]
        """
        df = crosstab_df.reset_index()
        df.columns = [sep.join(map(str, col)).strip(sep) for col in df.columns]
        index_col = index_col_name or crosstab_df.index.name
        df_melted = df.melt(id_vars=[index_col], var_name='Group', value_name='count')
        if df_melted['Group'].str.contains(sep).any():
            group_cols = ['Group_1', 'Group_2']
            df_melted[group_cols] = df_melted['Group'].str.split(sep, expand=True)
        else:
            group_cols = []
        return df_melted[[index_col, 'Group', 'count'] + group_cols]

    # Notebook cell 54
    flattened_results = {}
    for (key, df) in crosstab_results.items():
        flat_df = flatten_crosstab_simple(df, index_col_name=key)
        flattened_results[key] = flat_df

    # Notebook cell 55
    def plot_horizontal_stacked(df_flat, category_col, adhd_col='Group_2', sex_col='Group_1', wrap_width=40, height_per_row=40, title=None):
        """
        Plot a horizontal stacked bar chart showing ADHD vs Non, stacked by Sex.

        Parameters
        ----------
        df_flat : pd.DataFrame
            The melted/flattened crosstab DataFrame with columns like: [category_col, Group, count, Group_1, Group_2]

        category_col : str
            Name of the column with the categories (e.g., 'Barratt_Barratt_P1_Edu')

        adhd_col : str
            Column name that holds ADHD grouping (default: 'Group_2')

        sex_col : str
            Column name that holds Sex grouping (default: 'Group_1')

        wrap_width : int, default=40
            Number of characters to wrap category labels for readability

        height_per_row : int, default=40
            Height allocated per category (affects figure height)

        title : str, optional
            Title of the chart. If None, generated from column name.
        """
        df_flat['wrapped'] = df_flat[category_col].apply(lambda x: '<br>'.join(textwrap.wrap(str(x), width=wrap_width)))
        fig = px.bar(df_flat, x='count', y='wrapped', color=sex_col, orientation='h', facet_col=adhd_col, title=title or f'{category_col}: ADHD vs Non, stacked by sex', labels={'wrapped': category_col})
        fig.update_layout(height=max(400, height_per_row * df_flat[category_col].nunique()), margin=dict(l=300, r=30, t=50, b=50), yaxis=dict(tickfont=dict(size=10)), barmode='stack')
        report.figure(fig)

    # Notebook cell 56
    for (key, df) in flattened_results.items():
        df['wrapped'] = wrap_labels_html(df[key])
        fig = px.bar(df, x='count', y='wrapped', color='Group_1', facet_col='Group_2', orientation='h', title=f'{field_to_description.get(key, key)}: ADHD vs Non, stacked by sex', labels={'wrapped': field_to_description.get(key, key)})
        fig.update_layout(height=max(700, 35 * df[key].nunique()), margin=dict(l=20, r=20, t=50, b=50), yaxis=dict(tickfont=dict(size=10)), barmode='stack')
        report.figure(fig)
