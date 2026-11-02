"""Connectome EDA adapted from notebook cells 164–216. No model training."""

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
import networkx as nx
import community.community_louvain as community_louvain


def run(df_m, df_s, dictionary, report):
    display = report.display

    Markdown = str

    # Notebook cell 164
    df = df_m.merge(df_s, on='participant_id', how='inner').drop('participant_id', axis=1)

    # Notebook cell 165
    display(df.shape)

    # Notebook cell 166
    display(df.isnull().sum().sum())

    # Notebook cell 167
    display(df.columns)

    # Notebook cell 168
    feature_cols = [col for col in df.columns if col not in ['ADHD_Outcome', 'Sex_F']]

    # Notebook cell 169
    throw_column_map = {}
    for col in df.columns:
        match = re.match('(\\d+)throw_(\\d+)thcolumn', col)
        if match:
            throw = int(match.group(1))
            column = int(match.group(2))
            throw_column_map[throw, column] = col

    # Notebook cell 170
    adhd_female_mean = df[(df['ADHD_Outcome'] == 1) & (df['Sex_F'] == 1)][feature_cols].mean()
    adhd_male_mean = df[(df['ADHD_Outcome'] == 1) & (df['Sex_F'] == 0)][feature_cols].mean()
    other_female_mean = df[(df['ADHD_Outcome'] == 0) & (df['Sex_F'] == 1)][feature_cols].mean()
    other_male_mean = df[(df['ADHD_Outcome'] == 0) & (df['Sex_F'] == 0)][feature_cols].mean()

    # Notebook cell 171
    def build_feature_matrix(row, throw_column_map=throw_column_map, fill_value=np.nan, matrix_size=(200, 200)):
        """
        Build a (throw, column) matrix from one sampled row or mean vector,
        symmetrize it, and set the diagonal to 1.

        Parameters:
        - row: DataFrame (1 row) or Series (mean vector)
        - throw_column_map: dict {(throw, column): original column name}
        - fill_value: value to fill missing entries (default np.nan)
        - matrix_size: tuple (n_throws, n_columns)

        Returns:
        - A 2D NumPy array representing the symmetric feature matrix
          with 1s on the diagonal.
        """
        matrix = np.full(matrix_size, fill_value, dtype=float)
        for ((throw, column), col_name) in throw_column_map.items():
            if col_name in row.index:
                value = row[col_name]
                if isinstance(value, np.ndarray):
                    value = value[0]
                matrix[throw, column] = value
        trans = matrix.T
        matrix = np.where(np.isnan(matrix) & ~np.isnan(trans), trans, matrix)
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 1.0)
        return matrix

    # Notebook cell 172
    adhd_female_matrix = build_feature_matrix(adhd_female_mean)
    adhd_male_matrix = build_feature_matrix(adhd_male_mean)
    other_female_matrix = build_feature_matrix(other_female_mean)
    other_male_matrix = build_feature_matrix(other_male_mean)

    # Notebook cell 173
    diff_ADHD_vs_Other_F = adhd_female_matrix - other_female_matrix
    diff_ADHD_vs_Other_M = adhd_male_matrix - other_male_matrix
    diff_Female_vs_Male_A = adhd_female_matrix - adhd_male_matrix
    diff_Female_vs_Male_O = other_female_matrix - other_male_matrix

    display("## Heatmap for the difference of subgroups' mean")

    # Notebook cell 175
    (fig, axes) = plt.subplots(4, 1, figsize=(6, 20))
    for (ax, (matrix, title)) in zip(axes.flatten(), [(diff_ADHD_vs_Other_F, 'female group, adhd vs non-adhd'), (diff_ADHD_vs_Other_M, 'male group, adhd vs non-adhd'), (diff_Female_vs_Male_A, 'adhd group female vs male'), (diff_Female_vs_Male_O, 'non-adhd group female va male')]):
        mask = np.tril(np.ones_like(matrix, dtype=bool))
        sns.heatmap(matrix, mask=mask, square=False, linecolor='grey', center=0, ax=ax, vmin=-0.1, vmax=0.1)
        ax.set_title(f'In {title}', fontsize=14)
        ax.set_xlabel('Column', fontsize=10)
        ax.set_ylabel('Throw', fontsize=10)
        ax.set_xticks(np.arange(0, 200, 10))
        ax.set_yticks(np.arange(0, 200, 10))
    plt.tight_layout()
    report.figure()

    display('#### What is it:')

    display('## Network plot with difference between sub groups')

    display('### Terms:')

    # Notebook cell 179
    def build_connectivity_graph_plotly_density(matrix, density=0.05, title='Functional Connectivity Graph'):
        """
        Build and plot an interactive connectivity graph using Plotly,
        keeping the top `density` fraction of edges by absolute weight,
        and coloring nodes by community structure (Louvain method).

        Parameters:
        - matrix: 2D NumPy array (e.g., a correlation or difference matrix), square of size N×N.
        - density: fraction of all possible edges to keep (0 < density <= 1).
        - title: title for the graph.
        """
        import numpy as np
        import networkx as nx
        import plotly.graph_objects as go
        import plotly.express as px
        from community import community_louvain
        if matrix is None:
            print('No matrix provided.')
            return
        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
            print("Input 'matrix' must be a 2D NumPy array.")
            return
        if matrix.shape[0] != matrix.shape[1]:
            print(f'Warning: Matrix is not square {matrix.shape} – node labels may be confusing.')
        n_nodes = matrix.shape[0]
        abs_weights = [abs(matrix[i, j]) for i in range(n_nodes) for j in range(i + 1, n_nodes) if not np.isnan(matrix[i, j])]
        n_possible = len(abs_weights)
        if not n_possible:
            raise ValueError('No finite connectome edges available for this comparison')
        n_keep = max(1, int(np.floor(n_possible * density)))
        sorted_w = sorted(abs_weights, reverse=True)
        cutoff = sorted_w[n_keep - 1]
        G = nx.Graph()
        G.add_nodes_from((str(i) for i in range(n_nodes)))
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                w = matrix[i, j]
                if not np.isnan(w) and abs(w) >= cutoff:
                    G.add_edge(str(i), str(j), weight=abs(w), actual_diff=w)
        if G.number_of_edges() == 0:
            print(f'No edges kept at density={density:.3f} (cutoff={cutoff:.3f}).')
        else:
            print(f'Keeping {G.number_of_edges()} edges (top {density * 100:.1f}% by |weight|, cutoff={cutoff:.3f})')
        partition = {}
        node_colors = ['lightgray'] * n_nodes
        num_comms = 0
        try:
            partition = community_louvain.best_partition(G, weight='weight', random_state=42)
            num_comms = len(set(partition.values()))
            print(f'Detected {num_comms} communities.')
            palette = px.colors.qualitative.Plotly
            if num_comms > len(palette):
                repeats = num_comms // len(palette) + 1
                palette = palette * repeats
            for (node, comm) in partition.items():
                node_colors[int(node)] = palette[comm]
        except Exception as e:
            print(f'Community detection failed: {e}. All nodes will be gray.')
        try:
            pos = nx.spring_layout(G, weight='weight', k=1.0 / np.sqrt(n_nodes), iterations=50, seed=42)
        except:
            pos = nx.random_layout(G, seed=42)
        (edge_x, edge_y, edge_text) = ([], [], [])
        for (u, v, data) in G.edges(data=True):
            (x0, y0) = pos[u]
            (x1, y1) = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            txt = f"{u}–{v}<br>Δr = {data['actual_diff']:.3f}"
            edge_text += [txt, txt, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='text', hovertext=edge_text)
        (node_x, node_y, node_text, node_label) = ([], [], [], [])
        for i in range(n_nodes):
            (xi, yi) = pos[str(i)]
            node_x.append(xi)
            node_y.append(yi)
            deg = G.degree(str(i))
            comm = partition.get(str(i), -1)
            node_text.append(f'Node {i}<br>Degree: {deg}<br>Community: {comm}')
            node_label.append(str(i))
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_label, textposition='bottom center', hoverinfo='text', hovertext=node_text, marker=dict(size=10, color=node_colors, line=dict(width=1, color='black')))
        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title=f'{title} – Top {density * 100:.1f}% Edges, {num_comms} Communities', showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        report.figure(fig)
        return G

    # Notebook cell 180
    G1 = build_connectivity_graph_plotly_density(diff_ADHD_vs_Other_F, density=0.05, title='Female group: ADHD vs NON-ADHD')

    # Notebook cell 181
    G2 = build_connectivity_graph_plotly_density(diff_ADHD_vs_Other_M, density=0.05, title='Male group: ADHD vs NON-ADHD')

    # Notebook cell 182
    G3 = build_connectivity_graph_plotly_density(diff_Female_vs_Male_A, density=0.05, title='ADHD group: Female vs Male')

    # Notebook cell 183
    G4 = build_connectivity_graph_plotly_density(diff_Female_vs_Male_O, density=0.05, title='Non-ADHD group: Female vs Male')

    display('### Interpreting & Comparing Density-Based Graphs (Top 5% Edges)')

    display('## Comparing Modularity _Q_ Across Graphs')

    # Notebook cell 186
    graphs = {'Female ADHD vs Control': G1, 'Male ADHD vs Control': G2, 'ADHD: Female vs Male': G3, 'Control: Female vs Male': G4}
    records = []
    for (name, G) in graphs.items():
        partition = community_louvain.best_partition(G, weight='weight', random_state=42)
        Q = community_louvain.modularity(partition, G, weight='weight')
        records.append({'Comparison': name, 'Modularity_Q': Q})
    df_Q = pd.DataFrame(records)
    display(df_Q)

    display('#### Observe')

    display('### Summary ')

    display('### Summarize communites')

    # Notebook cell 190
    def summarize_communities(graphs, big_threshold=10):
        """
        Summarize the number of big communities and singleton communities in each graph.

        Parameters:
        - graphs: dict, {name: NetworkX graph}
        - big_threshold: int, minimum number of nodes to call a community 'big'

        Returns:
        - pandas DataFrame with summary
        """
        records = []
        for (name, G) in graphs.items():
            partition = community_louvain.best_partition(G, weight='weight', random_state=42)
            communities = {}
            for (node, comm) in partition.items():
                communities.setdefault(comm, []).append(node)
            num_big = 0
            num_singleton = 0
            for members in communities.values():
                size = len(members)
                if size >= big_threshold:
                    num_big += 1
                if size == 1:
                    num_singleton += 1
            records.append({'Comparison': name, 'Total_Communities': len(communities), 'Big_Communities': num_big, 'Singleton_Communities': num_singleton})
        df_summary = pd.DataFrame(records)
        return df_summary

    # Notebook cell 191
    df_comm_summary = summarize_communities(graphs, big_threshold=10)
    display(df_comm_summary)

    # Notebook cell 192
    df_plot = df_comm_summary.copy()
    df_plot['Other_Communities'] = df_plot['Total_Communities'] - df_plot['Big_Communities'] - df_plot['Singleton_Communities']
    df_plot.set_index('Comparison')[['Big_Communities', 'Singleton_Communities', 'Other_Communities']].plot(kind='bar', stacked=True, figsize=(7, 4), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('Number of Communities')
    plt.title('Community Size Breakdown by Comparison')
    plt.xticks(rotation=30, ha='right')
    plt.legend(['Big (≥10 nodes)', 'Singleton (1 node)', 'Other (2-9 nodes)'])
    plt.tight_layout()
    report.figure()

    display('#### Observe:')

    display('## Extracting node names inside big communities')

    # Notebook cell 195
    def extract_big_communities(graphs, big_threshold=10):
        """
        Extract node IDs for big communities (size ≥ threshold) from multiple graphs.

        Parameters:
        - graphs: dict, {name: NetworkX graph}
        - big_threshold: int, minimum number of nodes to consider a 'big' community.

        Returns:
        - dict: {Comparison name: {Community ID: list of node IDs}}
        """
        big_communities = {}
        for (name, G) in graphs.items():
            partition = community_louvain.best_partition(G, weight='weight', random_state=42)
            communities = {}
            for (node, comm_id) in partition.items():
                communities.setdefault(comm_id, []).append(node)
            big_comms = {}
            for (comm_id, members) in communities.items():
                if len(members) >= big_threshold:
                    big_comms[comm_id] = sorted(members)
            big_communities[name] = big_comms
        return big_communities

    # Notebook cell 196
    big_comms = extract_big_communities(graphs, big_threshold=10)

    # Notebook cell 197
    def compute_community_features(matrix, communities):
        """
        For a given subject (or group average) matrix and community dict,
        compute mean intra-community and inter-community connectivity.

        Parameters:
        - matrix: 2D NumPy array (connectivity or difference matrix)
        - communities: dict {community_id: list of node IDs (strings)}

        Returns:
        - dict of features {feature_name: value}
        """
        features = {}
        comm_ids = list(communities.keys())
        for cid in comm_ids:
            nodes = [int(n) for n in communities[cid]]
            if len(nodes) < 2:
                features[f'intra_comm_{cid}'] = np.nan
                continue
            values = []
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    v = matrix[nodes[i], nodes[j]]
                    if not np.isnan(v):
                        values.append(v)
            features[f'intra_comm_{cid}'] = np.mean(values) if values else np.nan
        for i in range(len(comm_ids)):
            for j in range(i + 1, len(comm_ids)):
                (cid1, cid2) = (comm_ids[i], comm_ids[j])
                nodes1 = [int(n) for n in communities[cid1]]
                nodes2 = [int(n) for n in communities[cid2]]
                values = []
                for n1 in nodes1:
                    for n2 in nodes2:
                        v = matrix[n1, n2]
                        if not np.isnan(v):
                            values.append(v)
                features[f'inter_comm_{cid1}_{cid2}'] = np.mean(values) if values else np.nan
        return features

    # Notebook cell 198
    features_ADHD_vs_Other_F = compute_community_features(diff_ADHD_vs_Other_F, big_comms['Female ADHD vs Control'])
    features_ADHD_vs_Other_M = compute_community_features(diff_ADHD_vs_Other_M, big_comms['Male ADHD vs Control'])
    features_Female_vs_Male_A = compute_community_features(diff_Female_vs_Male_A, big_comms['ADHD: Female vs Male'])
    features_Female_vs_Male_O = compute_community_features(diff_Female_vs_Male_O, big_comms['Control: Female vs Male'])

    # Notebook cell 199
    df_features = pd.DataFrame([{**features_ADHD_vs_Other_F, 'Comparison': 'Female ADHD vs Control'}, {**features_ADHD_vs_Other_M, 'Comparison': 'Male ADHD vs Control'}, {**features_Female_vs_Male_A, 'Comparison': 'ADHD: Female vs Male'}, {**features_Female_vs_Male_O, 'Comparison': 'Control: Female vs Male'}])

    # Notebook cell 200
    if not any((col.startswith('intra_comm_') for col in df_features.columns)):
        display('No communities with at least 10 nodes; skipping community feature analysis.')
        return
    selected_columns = [col for col in df_features.columns if col.startswith('intra_comm_')]
    df_features.set_index('Comparison')[selected_columns].T.plot(kind='bar', figsize=(8, 5))
    plt.title('Intra-Community Connectivity Differences')
    plt.ylabel('Mean connectivity difference')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()
    report.figure()

    display('#### Obervation:')

    display('### Spider plot for intra-community features')

    # Notebook cell 203
    df_norm = df_features.copy()
    selected_columns = [col for col in df_norm.columns if col.startswith('intra_comm_')]
    max_abs = df_norm[selected_columns].abs().max().max()
    df_norm[selected_columns] = df_norm[selected_columns] / (max_abs if max_abs > 0 else 1)

    # Notebook cell 204
    num_vars = len(selected_columns)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    (fig, ax) = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for (idx, row) in df_norm.iterrows():
        values = row[selected_columns].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row['Comparison'])
        ax.fill(angles, values, alpha=0.1)
    ax.plot(angles, [0] * len(angles), color='black', linewidth=2, linestyle='--')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), selected_columns)
    ax.set_ylim(-1, 1)
    ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=0.7)
    ax.xaxis.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Normalized Intra-Community Connectivity Differences (Radar Plot)', y=1.1)
    report.figure()

    display('#### Interpretation:')

    display('## Create graph features Dataframe')

    # Notebook cell 207
    subject_features = []
    for (idx, row) in df.iterrows():
        subject_matrix = build_feature_matrix(row)
        features = compute_community_features(subject_matrix, big_comms['Female ADHD vs Control'])
        features['participant_id'] = df_m.iloc[idx]['participant_id']
        features['ADHD_Outcome'] = row['ADHD_Outcome']
        features['Sex_F'] = row['Sex_F']
        subject_features.append(features)
    df_subject_features = pd.DataFrame(subject_features)

    # Notebook cell 208
    df_subject_features.to_csv(report.directory / 'subject_graph_features.csv', index=False)

    # Notebook cell 209
    display(df_subject_features.shape)

    # Notebook cell 210
    display(df_subject_features.columns)

    display('## Statistical Testing on graph features')

    # Notebook cell 212
    def compute_pvalues(df_subject_features, group_column, group1_value, group2_value):
        """
        Perform T-tests between two groups on intra- and inter-community features.

        Parameters:
        - df_subject_features: DataFrame with subject features and labels
        - group_column: Name of the column to split groups (e.g., 'Sex_F', 'ADHD_Outcome')
        - group1_value: Value corresponding to first group (e.g., 1)
        - group2_value: Value corresponding to second group (e.g., 0)

        Returns:
        - df_pvals: DataFrame with Feature, p_value, and Significance
        """
        selected_columns = [col for col in df_subject_features.columns if col.startswith('intra_comm') or col.startswith('inter_comm')]
        group1 = df_subject_features[df_subject_features[group_column] == group1_value]
        group2 = df_subject_features[df_subject_features[group_column] == group2_value]
        p_values = []
        for feature in selected_columns:
            (stat, pval) = ttest_ind(group1[feature], group2[feature], equal_var=False, nan_policy='omit')
            p_values.append((feature, pval))
        df_pvals = pd.DataFrame(p_values, columns=['Feature', 'p_value']).sort_values('p_value')

        def significance(p):
            if p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            else:
                return ''
        df_pvals['Significance'] = df_pvals['p_value'].apply(significance)
        return df_pvals

    # Notebook cell 213
    df_pvals_sex = compute_pvalues(df_subject_features, group_column='Sex_F', group1_value=1, group2_value=0)
    df_pvals_adhd = compute_pvalues(df_subject_features, group_column='ADHD_Outcome', group1_value=1, group2_value=0)

    # Notebook cell 214
    display(df_pvals_sex)

    display('#### Statistically Important Features for Sex ')

    # Notebook cell 216
    display(df_pvals_adhd)
