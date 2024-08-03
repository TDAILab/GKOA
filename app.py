import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_distances
import plotly.graph_objects as go
import streamlit as st


# カスタムCSSを更新して、テーブルの装飾を追加
st.markdown("""
<style>
    /* Overall style */
    .reportview-container {
        background-color: #FFFFFF;
        font-family: 'Helvetica', 'Arial', sans-serif;
    }
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1280px;
        margin: 0 auto;
    }
    

    /* Typography */
    h2 {
        color: #091F7C;
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    h3 {
        color: #6366F1;
        font-size: 1.4rem;
        font-weight: 600;
    }
    p, .streamlit-expanderHeader {
        color: #4B5563;
        font-size: 1rem;
        line-height: 1.5;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #F3F4F6;
        padding: 2rem 1rem;
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #ffffff;
        border: 1px solid #D1D5DB;
        border-radius: 4px;
        padding: 0.5rem;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-size: 1.2em;
        color: #4F46E5;
        background-color: #EEF2FF;
        border: none;
        border-radius: 4px;
    }
    .streamlit-expanderContent {
        border: 1px solid #C7D2FE;
        border-top: none;
        border-radius: 0 0 4px 4px;
    }

    /* Table styles */
    .dataframe {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        font-family: sans-serif;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    .dataframe thead tr {
        background-color: #4F46E5;
        color: #ffffff;
        text-align: left;
    }
    .dataframe th,
    .dataframe td {
        padding: 12px 15px;
    }
    .dataframe tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    .dataframe tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .dataframe tbody tr:last-of-type {
        border-bottom: 2px solid #4F46E5;
    }
</style>
""", unsafe_allow_html=True)

SIMILARYTY_THRESHOLD = 0.5


def find_closest_name(cluster_num, embeddings, center, names):
    distances = cosine_distances(np.stack(embeddings), center.reshape(1, -1))
    closest_index = np.argmin(distances)
    return names.iloc[closest_index]



def reanalyze_with_summary(summary):
    # df['mentioned_in_summary'] = df.apply(lambda row: 
    #     row['source'].lower() in summary.lower() or 
    #     row['target'].lower() in summary.lower(), axis=1)
    # df["success"] = True
    return summary

def save_summary(summary):
    input_folder = "input"
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    
    with open(os.path.join(input_folder, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

def get_row_color(weight, min_weight, max_weight, cmap, alpha=0.3):
    normalized = (weight - min_weight) / (max_weight - min_weight)
    rgba = cmap(normalized)
    hex_color = 'rgba({},{},{},{})'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), alpha)
    return f'background-color: {hex_color}'

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

new_cmap = truncate_colormap(plt.get_cmap('jet'), 1.0, 0.45)

def load_data_from_output_folder(folder_path):
    entities_path = os.path.join(folder_path, "artifacts", "create_final_entities.parquet")
    documents_path = os.path.join(folder_path, "artifacts", "create_final_documents.parquet")
    
    if os.path.exists(entities_path):
        entities = pd.read_parquet(entities_path)
        
        if os.path.exists(documents_path):
            entities = entities.explode("text_unit_ids")
            documents = pd.read_parquet(documents_path).explode("text_unit_ids")
            
            # Merge entities with documents
            entities = entities.merge(documents[['text_unit_ids', 'raw_content']], on='text_unit_ids', how='left')
            
            # Replace text_unit_ids with raw_content
            entities['text_unit_ids'] = entities['raw_content'].fillna(entities['text_unit_ids'])
            
            # Drop the raw_content column as it's no longer needed
            entities = entities.drop(columns=['raw_content'])
        
        return entities
    else:
        return None

def get_output_folders(output_dir="output"):
    folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    return sorted(folders, reverse=True)

def create_network_graph(G, highlight_nodes=[], centrality_measure='PageRank'):
    sources = st.session_state.entities_raw['source'].values
    text_unit_ids = st.session_state.entities_raw['text_unit_ids'].values
    
    if 'positions' in st.session_state:
        positions = st.session_state.positions
    else:
        positions = nx.spring_layout(G)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#D3D3D3'),
        hoverinfo='none',
        mode='lines')
    
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        
        if node in highlight_nodes:
            node_colors.append('#A6E2B5')  # ハイライト色
        elif node in sources:
            node_colors.append('red')
        elif node in text_unit_ids:
            node_colors.append('#FCE724')
        else:
            node_colors.append('blue')
      
        if node in st.session_state.entities['source'].values:
            centrality = st.session_state.entities[st.session_state.entities['source'] == node][centrality_measure].fillna(0).values[0]
            if centrality:
                node_sizes.append(10 + 100 * centrality)  # サイズをスケール
            else:
                node_sizes.append(10)  # デフォルトサイズ
                
        else:
            node_sizes.append(10)  # デフォルトサイズ
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line_width=1))
    
    node_adjacencies = []
    node_text = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
        node_text.append(f'Node: {node}<br>Connections: {len(adjacencies)}')
    
    node_trace.text = node_text
    
    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        height=500, 
        width=600,  
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    
    return fig

def analyze_parquet_data(entities):
    # Filter entities
    entities = entities[entities["type"] == "REASON"]

    # Perform hierarchical clustering
    Z = linkage(np.stack(entities.name_embedding), method='average', metric="cosine")
    threshold = 0.4
    clusters = fcluster(Z, threshold, criterion='distance')     
    entities["cluster"] = clusters      

    # Calculate cluster centers
    cluster_centers = entities.groupby('cluster')['name_embedding'].apply(lambda x: np.mean(np.stack(x), axis=0))

    # Find closest name for each cluster
    new_names = entities['cluster'].apply(lambda cluster_num: find_closest_name(
        cluster_num, 
        entities[entities['cluster'] == cluster_num]['description_embedding'].values, 
        cluster_centers[cluster_num], 
        entities[entities["cluster"] == cluster_num]["name"]
    ))

    entities["source"] = new_names 

    # Explode text_unit_ids
    entities = entities.explode("text_unit_ids")
    
    # Create network graph
    G = nx.from_pandas_edgelist(entities, source="source", target="text_unit_ids", create_using=nx.Graph())

    # Calculate PageRank
    pagerank = nx.pagerank(G)
    entities["PageRank"] = entities["source"].map(pagerank)
    if not 'sort_by' in st.session_state:
        st.session_state.sort_by = "PageRank"

    entities_raw = entities.copy()

    entities = entities.drop_duplicates(subset=["source"], keep="first")
    
    # Sort and reset index
    entities = entities.sort_values("PageRank", ascending=False).reset_index(drop=True)

    return entities, G, entities_raw



def load_summary_entities(folder_path):
    summary_entities_path = os.path.join(folder_path, "artifacts", "create_final_entities.parquet")
    if os.path.exists(summary_entities_path):
        return pd.read_parquet(summary_entities_path)
    else:
        return None


def create_similarity_heatmap(entities, summary_entities):
    similarities = []
    for _, entity in entities.iterrows():
        entity_similarities = []
        for _, summary_entity in summary_entities.iterrows():
            sim = calculate_cosine_similarity(entity['name_embedding'], summary_entity['name_embedding'])
            entity_similarities.append(sim)
        similarities.append(entity_similarities)

    fig = go.Figure(data=go.Heatmap(
        z=similarities,
        x=summary_entities['name'],
        y=entities['name'],
        colorscale='Viridis',
        colorbar=dict(title='Cosine Similarity')
    ))

    fig.update_layout(
        title='Cosine Similarity Heatmap',
        xaxis_title='Summary Entities',
        yaxis_title='Original Entities',
        height=800,
        width=1000
    )

    return fig



def process_csv(csv_file, summary_text=None):
    input_folder = "input"
    if os.path.exists(input_folder):
        shutil.rmtree(input_folder)
    os.makedirs(input_folder)

    if csv_file:
        df = pd.read_csv(csv_file)
        
        for index, row in df.iterrows():
            with open(os.path.join(input_folder, f"text_{index}.txt"), "w", encoding="utf-8") as f:
                f.write(row['text'])
    elif summary_text:
        with open(os.path.join(input_folder, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(summary_text)

    # graphrag.indexを実行
    try:
        result = subprocess.run(["python", "-m", "graphrag.index", "--root", "."], 
                                capture_output=True, text=True, check=True)
        st.success("Graph creation completed successfully.")

        # summary_textがある場合（summary_inputから来た場合）のみ
        if summary_text:
            latest_folder = get_latest_output_folder()
            if latest_folder:
                src_folder = os.path.join("output", latest_folder)
                dst_folder = os.path.join("output_summary", latest_folder)
                                
                # 最新の出力フォルダの内容をoutput_summaryにコピー
                shutil.copytree(src_folder, dst_folder)
                st.success(f"Copied latest output to {dst_folder}")
                
                # output_summaryフォルダが存在する場合は削除
                if os.path.exists(src_folder):
                    shutil.rmtree(src_folder)

        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error occurred while creating graph: {e.stderr}")
        return False

def get_latest_output_folder(output_dir = "output"):
    folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    return max(folders, key=lambda x: os.path.getctime(os.path.join(output_dir, x))) if folders else None

def main():
    # Sidebar
    with st.sidebar:
        st.markdown(
            """
            <style>
            [data-testid=stSidebar] {
                background-color: #F2F5F9;
            }
            .sidebar-title {
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 0px;
            }
            .sidebar-title img {
                width: 50px;
                height: 50px;
                margin-right: 10px;
            }
            .sidebar-title h1 {
                color: #0D29AB;
                font-size: 24px;
            }
            </style>
            <div class="sidebar-title">
                <h1>Graph-based Key Opinion Analyser (GKOA)</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )

        st.header("Data Input")
        input_option = st.radio("Choose input method:", ["Upload CSV", "Select from Output Folder"])

        if input_option == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV file", type="csv")
            if uploaded_file is not None:
                st.session_state.uploaded_file = uploaded_file
                st.success("CSV file uploaded successfully.")
                
                if st.button("Analyze"):
                    with st.spinner("Creating graph..."):
                        if process_csv(uploaded_file):
                            st.session_state.graph_created = True
                            # グラフ作成後、最新の出力フォルダを自動的に選択
                            latest_folder = get_latest_output_folder("output")
                            if latest_folder:
                                st.session_state.selected_folder = latest_folder
                        else:
                            st.session_state.graph_created = False
        
        output_folders = get_output_folders("output")
        if input_option == "Select from Output Folder" or st.session_state.get('graph_created', False):
            selected_folder = st.selectbox("Select output folder:", output_folders, 
                                           index=output_folders.index(st.session_state.get('selected_folder', output_folders[0])) if st.session_state.get('selected_folder') in output_folders else 0)
        
            if selected_folder:
                folder_path = os.path.join("output", selected_folder)
                entities = load_data_from_output_folder(folder_path)
                if entities is not None:
                    st.session_state.entities = entities
                    st.session_state.input_changed = True
                    st.success(f"Data loaded from {selected_folder}")
                else:
                    st.error("Required files not found in the selected folder.")


        st.header("Summary Input")
        summary_input_option = st.radio("Choose summary input method:", ["Input Box", "Select from Output Folder"])

        
        if summary_input_option == "Input Box":
            summary_text = st.text_area("Enter summary text:", height = 300, value = "The comments overwhelmingly express opposition to mandatory childhood vaccinations. Key reasons include concerns about potential side effects, unproven effectiveness, and religious beliefs. Many argue that children have strong immune systems and do not significantly spread the virus, making vaccinations unnecessary. The decision should lie with parents, respecting their right to choose and protect their children based on personal or religious beliefs. Some fear that the vaccines could cause severe health issues, including autism or other long-term problems. Others stress the need for more studies on vaccine safety and effectiveness before making them compulsory.")
            if summary_text:
                st.session_state.summary_text = summary_text
                
                # Reanalyzeボタンを追加
                if st.button("Reanalyze"):
                    with st.spinner("Reanalyzing..."):
                        if process_csv(None, summary_text):
                            st.session_state.graph_created = True
                            # 最新の出力フォルダを自動的に選択（Summary Input用）
                            latest_folder = get_latest_output_folder("output_summary")
                            if latest_folder:
                                st.session_state.summary_folder = latest_folder
                                # summary_entitiesを読み込む
                                summary_folder_path = os.path.join("output_summary", latest_folder)
                                summary_entities = load_summary_entities(summary_folder_path)
                                if summary_entities is not None:
                                    st.session_state.summary_entities = summary_entities
                                    st.session_state.summary_changed = True
                                    st.success(f"Reanalysis completed. Summary data updated from {latest_folder}")
                                else:
                                    st.warning(f"No entities found in {latest_folder}")
                        else:
                            st.error("Error occurred during reanalysis.")
        else:
            output_folders = get_output_folders("output_summary")
            selected_summary_folder = st.selectbox("Select output folder for summary:", output_folders, key="summary_folder")
            if selected_summary_folder:
                summary_folder_path = os.path.join("output_summary", selected_summary_folder)
                summary_entities = load_summary_entities(summary_folder_path)
                if summary_entities is not None:
                    st.session_state.summary_entities = summary_entities
                    st.session_state.summary_changed = True
                    st.success(f"Summary entities loaded from {selected_summary_folder}")
                else:
                    st.warning(f"No entities found in {selected_summary_folder}")

    # Main area
    if st.session_state.get('input_changed', False) or st.session_state.get('summary_changed', False):
        with st.spinner('Running analysis...'):
            if 'data' in st.session_state:
                # CSVデータの分析を行う（元のコードの分析ロジックを維持）
                entities = st.session_state.data  # この部分は元の分析ロジックに置き換える
            elif 'entities' in st.session_state:
                # Parquetデータの分析を行う
                entities = st.session_state.entities
                entities, G, entities_raw = analyze_parquet_data(entities)
                
            if 'summary_entities' in st.session_state:
                # Re-analyze using the summary entities
                summary_entities = st.session_state.summary_entities
                
                similarity_heatmap = 1-cosine_distances(
                    np.stack(entities["name_embedding"]),
                    np.stack(summary_entities["name_embedding"])
                    )
                score = similarity_heatmap.max(axis=1)

                idx = similarity_heatmap.argmax(axis=1)
                entities["most_similar_summary"] = summary_entities.iloc[idx]["name"].values
                entities['similarity'] = score
                st.session_state.entities = entities

            st.session_state.entities = entities
            st.session_state.G = G
            st.session_state.entities_raw = entities_raw

            # Generate network graph
            highlight_entities = []
            if 'summary_entities' in st.session_state:
                if "similarity" in st.session_state.entities:
                    highlight_entities = st.session_state.entities[st.session_state.entities['similarity'] > SIMILARYTY_THRESHOLD]['source'].tolist()
                    
                # highlight_entities = st.session_state.summary_entities['name'].tolist()
            network_fig = create_network_graph(st.session_state.G, highlight_nodes=highlight_entities)
            st.session_state.network_fig = network_fig

            st.session_state.input_changed = False
            st.session_state.summary_changed = False
            st.success("Analysis completed.")

    if 'entities' in st.session_state:
        st.subheader("Analysis Results")


        # Add dropdown for sorting
        sort_options = ['PageRank', 'Betweenness', 'Closeness', 'Eigenvector', 'Degree']
        st.session_state.sort_options = sort_options

        sort_by = st.selectbox("Sort by:", sort_options)
        st.session_state.sort_by = sort_by

        if sort_by not in st.session_state.entities.columns:  
            if sort_by == "PageRank":
                d = nx.pagerank(G)
                st.session_state.entities["PageRank"] = [d[node] for node in st.session_state.entities["source"]]
            elif sort_by == "Betweenness":
                d = nx.betweenness_centrality(G)
                st.session_state.entities["Betweenness"] = [d[node] for node in st.session_state.entities["source"]]
            elif sort_by == "Closeness":
                d = nx.closeness_centrality(G)
                st.session_state.entities["Closeness"] = [d[node] for node in st.session_state.entities["source"]]
            elif sort_by == "Eigenvector":
                d = nx.eigenvector_centrality(G)
                st.session_state.entities["Eigenvector"] = [d[node] for node in st.session_state.entities["source"]]
            elif sort_by == "Degree":
                d = nx.degree_centrality(G)
                st.session_state.entities["Degree"] = [d[node] for node in st.session_state.entities["source"]]

        
        st.session_state.entities = st.session_state.entities.sort_values(sort_by, ascending=False).reset_index(drop=True)
        
        
        min_weight = st.session_state.entities[sort_by].min()
        max_weight = st.session_state.entities[sort_by].max()
        
        def apply_row_colors(row):
            weight = row[sort_by]
            return [get_row_color(weight, min_weight, max_weight, cmap = cm.get_cmap('rainbow'), alpha=0.3)] * len(row)
        

        def apply_row_colors2(row):
            weight = row['similarity']
            return [get_row_color(weight, 0, 1, cmap = new_cmap ,alpha=0.3)] * len(row)


        columns_to_display = ['source', 'description', 'text_unit_ids']
        if 'summary_entities' in st.session_state:
            columns_to_display += [st.session_state.sort_by, 'similarity', 'most_similar_summary']
            st.session_state.entities_colored = st.session_state.entities[columns_to_display].style.apply(apply_row_colors2, axis=1)
            st.dataframe(st.session_state.entities_colored, use_container_width=True, height=250)
        else:
            columns_to_display += [st.session_state.sort_by]
            st.session_state.entities_colored = st.session_state.entities[columns_to_display].style.apply(apply_row_colors, axis=1)
            st.dataframe(st.session_state.entities_colored, use_container_width=True, height=250)

        st.subheader("Network Visualization")
        if 'network_fig' in st.session_state:
            # Create network graph based on selected centrality measure
            highlight_entities = []
            if 'summary_entities' in st.session_state:
                if "similarity" in st.session_state.entities:
                    highlight_entities = st.session_state.entities[st.session_state.entities['similarity'] > SIMILARYTY_THRESHOLD]['source'].tolist()
            network_fig = create_network_graph(st.session_state.G, highlight_nodes=highlight_entities, centrality_measure=sort_by)
            st.plotly_chart(network_fig, use_container_width=True, config={'displayModeBar': True})
        else:
            st.info("Network graph not available. Please run the analysis to generate the graph.")

    else:
        st.info("Please upload data or select an output folder to view results.")

if __name__ == "__main__":
    main()