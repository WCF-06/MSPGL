# MSPGL reviewer-facing code
# Exported from MSPGL.ipynb. Markdown cells are preserved as comments.

# %% [markdown] Cell 1
# # Build graph-structured data and visualize it
#
# Reviewer note: update only the path configuration blocks in each section if your data/model folders use different locations. The default paths are relative to the working directory.

# %% Cell 2
import torch
import os
import numpy as np
import pandas as pd
import logging
import networkx as nx
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import plotly.graph_objects as go
from torch_geometric.data import Data
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PerformanceTracker:
    """Record processing performance and device information."""
    def __init__(self):
        self.device_info = self._get_device_info()
        self.results = []
        
    def _get_device_info(self):
        """Get information about the active compute device."""
        device_info = {
            'device': 'CPU',
            'device_name': 'CPU',
            'is_cuda': False,
            'is_mps': False,
            'torch_version': torch.__version__
        }
        
        if torch.cuda.is_available():
            device_info['device'] = 'CUDA'
            device_info['device_name'] = torch.cuda.get_device_name(0)
            device_info['is_cuda'] = True
            device_info['cuda_version'] = torch.version.cuda
            device_info['num_gpus'] = torch.cuda.device_count()
            device_info['cuda_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_info['device'] = 'MPS'
            device_info['device_name'] = 'Apple Metal Performance Shaders'
            device_info['is_mps'] = True
        
        try:
            import platform
            import psutil
            device_info['platform'] = platform.platform()
            device_info['cpu_count'] = psutil.cpu_count()
            device_info['total_memory_gb'] = f"{psutil.virtual_memory().total / 1e9:.2f} GB"
            device_info['available_memory_gb'] = f"{psutil.virtual_memory().available / 1e9:.2f} GB"
        except ImportError:
            pass
        
        return device_info
    
    def log_device_info(self):
        """Log compute device information."""
        logging.info("\n" + "="*80)
        logging.info("Current Compute Device Information")
        logging.info("="*80)
        logging.info(f"Primary device: {self.device_info['device']}")
        logging.info(f"Device name: {self.device_info['device_name']}")
        logging.info(f"PyTorch version: {self.device_info['torch_version']}")
        
        if self.device_info['is_cuda']:
            logging.info(f"CUDA version: {self.device_info.get('cuda_version')}")
            logging.info(f"Number of GPUs: {self.device_info.get('num_gpus')}")
            logging.info(f"GPU memory: {self.device_info.get('cuda_memory')}")
        elif self.device_info['is_mps']:
            logging.info("MPS acceleration is available.")
        
        if 'platform' in self.device_info:
            logging.info(f"Platform: {self.device_info.get('platform')}")
            logging.info(f"CPU count: {self.device_info.get('cpu_count')}")
            logging.info(f"Total memory: {self.device_info.get('total_memory_gb')}")
            logging.info(f"Available memory: {self.device_info.get('available_memory_gb')}")
        
        logging.info("="*80 + "\n")
    
    def add_result(self, file_key, status, total_time, subfolder_name, 
                   num_nodes=None, num_edges=None, error_msg=None, 
                   feature_dim=None, stages_time=None):
        """
        Record the processing result for one source CSV file.
        
        Args:
            file_key: file identifier (subfolder/filename)
            status: processing status ('success', 'failed')
            total_time: total runtime in seconds
            subfolder_name: subfolder name
            num_nodes: number of nodes
            num_edges: number of edges
            error_msg: error message if processing fails
            feature_dim: feature dimension
            stages_time: per-stage timing dictionary with keys such as
                'csv_read', 'feature_calc', 'graph_build', 'visualization',
                'pyg_convert', 'density_calc', and 'save'
        """
        result = {
            'file_name': file_key,
            'subfolder': subfolder_name,
            'status': status,
            'total_time_s': round(total_time, 2),
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'feature_dimension': feature_dim,
            'error_msg': error_msg if error_msg else '',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if stages_time:
            for stage, stage_time in stages_time.items():
                result[f'time_{stage}'] = round(stage_time, 3)
        
        self.results.append(result)
    
    def save_results_to_csv(self, output_dir):
        """Save processing results as CSV and HTML reports."""
        if not self.results:
            logging.warning("No processing results are available; report files were not generated.")
            return
        
        df = pd.DataFrame(self.results)
        
        success_count = len(df[df['status'] == 'success'])
        failed_count = len(df[df['status'] == 'failed'])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_path = os.path.join(output_dir, f'processing_performance_{timestamp}.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"Detailed processing results saved to: {csv_path}")
        
        summary_data = {
            'Metric': [],
            'Value': []
        }

        summary_data['Metric'].extend([
            'Total files',
            'Successful files',
            'Failed files',
            'Success rate (%)',
            'Total time (s)',
            'Average time (s)',
            'Maximum time (s)',
            'Minimum time (s)',
            'Average node count',
            'Average edge count',
            'Average feature dimension'
        ])

        total_count = len(df)
        success_df = df[df['status'] == 'success']
        
        if len(success_df) > 0:
            avg_time = success_df['total_time_s'].mean()
            max_time = success_df['total_time_s'].max()
            min_time = success_df['total_time_s'].min()
            total_time = success_df['total_time_s'].sum()
            avg_nodes = success_df['num_nodes'].mean()
            avg_edges = success_df['num_edges'].mean()
            avg_feature_dim = success_df['feature_dimension'].mean()
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        else:
            avg_time = max_time = min_time = total_time = avg_nodes = avg_edges = avg_feature_dim = 0
            success_rate = 0
        
        summary_data['Value'].extend([
            total_count,
            success_count,
            failed_count,
            f"{success_rate:.1f}",
            f"{total_time:.2f}",
            f"{avg_time:.2f}",
            f"{max_time:.2f}",
            f"{min_time:.2f}",
            f"{avg_nodes:.0f}",
            f"{avg_edges:.0f}",
            f"{avg_feature_dim:.0f}"
        ])

        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(output_dir, f'processing_summary_{timestamp}.csv')
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"Processing summary saved to: {summary_csv_path}")
        
        self._generate_html_report(output_dir, timestamp, df, summary_df)
        
        self._print_console_summary(summary_df, df)
    
    def _generate_html_report(self, output_dir, timestamp, df, summary_df):
        """Generate a formatted HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Data Processing Performance Report</title>
            <style>
                body {{
                    font-family: 'Microsoft YaHei', Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 28px;
                }}
                .device-info {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                    border-left: 4px solid #667eea;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .device-info h3 {{
                    margin-top: 0;
                    color: #333;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }}
                .device-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 15px;
                    margin-top: 10px;
                }}
                .device-item {{
                    background: #f9f9f9;
                    padding: 10px;
                    border-radius: 4px;
                    border-left: 3px solid #667eea;
                }}
                .device-label {{
                    font-weight: bold;
                    color: #667eea;
                    font-size: 12px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                .device-value {{
                    color: #333;
                    margin-top: 5px;
                    font-size: 14px;
                }}
                .summary {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .summary h3 {{
                    margin-top: 0;
                    color: #333;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    border-radius: 8px;
                    overflow: hidden;
                }}
                table th {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 15px;
                    text-align: left;
                    font-weight: 600;
                    border: none;
                }}
                table td {{
                    padding: 12px 15px;
                    border-bottom: 1px solid #eee;
                }}
                table tbody tr:hover {{
                    background-color: #f9f9f9;
                }}
                table tbody tr:nth-child(even) {{
                    background-color: #fafafa;
                }}
                .status-success {{
                    color: #28a745;
                    font-weight: bold;
                }}
                .status-failed {{
                    color: #dc3545;
                    font-weight: bold;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }}
                .stat-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .stat-card.success {{
                    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                }}
                .stat-card.failed {{
                    background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
                }}
                .stat-value {{
                    font-size: 32px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .stat-label {{
                    font-size: 12px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    opacity: 0.9;
                }}
                .footer {{
                    text-align: center;
                    color: #999;
                    font-size: 12px;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Processing Performance Report</h1>
                <p>Generated automatically by the MSPGL pipeline.</p>
            </div>
            
            <div class="device-info">
                <h2>Compute Environment</h2>
                <div class="device-grid">
        """
        
        device_items = [
            ('Device item', self.device_info['device']),
            ('Device name', self.device_info['device_name']),
            ('PyTorch version', self.device_info['torch_version']),
        ]
        
        if self.device_info['is_cuda']:
            device_items.extend([
                ('CUDA version', self.device_info['cuda_version']),
                ('GPU count', str(self.device_info['num_gpus'])),
                ('GPU memory', self.device_info['cuda_memory']),
            ])
        
        if 'platform' in self.device_info:
            device_items.extend([
                ('System platform', self.device_info['platform']),
                ('CPU cores', str(self.device_info['cpu_count'])),
                ('Total memory', self.device_info['total_memory_gb']),
                ('Available memory', self.device_info['available_memory_gb']),
            ])
        
        for label, value in device_items:
            html_content += f"""
                    <div class="device-item">
                        <div class="device-label">{label}</div>
                        <div class="device-value">{value}</div>
                    </div>
            """
        
        html_content += """
                </div>
            </div>
            
            <div class="summary">
                <h2>Processing Summary</h2>
                <div class="stats-grid">
        """
        
        success_count = len(df[df['status'] == 'success'])
        failed_count = len(df[df['status'] == 'failed'])
        total_count = len(df)
        
        html_content += f"""
                    <div class="stat-card">
                        <div class="stat-label">Metric</div>
                        <div class="stat-value">{total_count}</div>
                    </div>
                    <div class="stat-card success">
                        <div class="stat-label">Metric</div>
                        <div class="stat-value">{success_count}</div>
                    </div>
                    <div class="stat-card failed">
                        <div class="stat-label">Metric</div>
                        <div class="stat-value">{failed_count}</div>
                    </div>
        """
        
        if success_count > 0:
            success_rate = (success_count / total_count * 100)
            html_content += f"""
                    <div class="stat-card success">
                        <div class="stat-label">Metric</div>
                        <div class="stat-value">{success_rate:.1f}%</div>
                    </div>
            """
        
        html_content += """
                </div>
                <table>
                    <thead>
                        <tr>
        """
        
        for col in summary_df.columns:
            html_content += f"<th>{col}</th>"
        
        html_content += """
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for _, row in summary_df.iterrows():
            html_content += "<tr>"
            for value in row:
                html_content += f"<td>{value}</td>"
            html_content += "</tr>"
        
        html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="summary">
                <h2>File-Level Processing Details</h2>
                <table>
                    <thead>
                        <tr>
        """
        
        for col in df.columns:
            html_content += f"<th>{col}</th>"
        
        html_content += """
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for _, row in df.iterrows():
            html_content += "<tr>"
            for col, value in row.items():
                if col == 'status':
                    status_class = 'status-success' if value == 'success' else 'status-failed'
                    html_content += f'<td class="{status_class}">{value}</td>'
                else:
                    html_content += f"<td>{value}</td>"
            html_content += "</tr>"
        
        html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                Generated by the MSPGL graph preprocessing pipeline.
            </div>
        </body>
        </html>
        """
        
        html_path = os.path.join(output_dir, f'processing_report_{timestamp}.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logging.info(f"HTML processing report saved to: {html_path}")
    
    def _print_console_summary(self, summary_df, df):
        """Print a concise console summary."""
        logging.info("\n" + "="*80)
        logging.info("Processing Summary")
        logging.info("="*80)
        
        for _, row in summary_df.iterrows():
            logging.info(f"{row['Metric']}: {row['Value']}")
        
        logging.info("="*80)
        
        success_df = df[df['status'] == 'success']
        if len(success_df) > 0:
            logging.info("Slowest successfully processed files:")
            slowest = success_df.nlargest(5, 'total_time_s')[['file_name', 'total_time_s', 'num_nodes', 'num_edges']]
            for idx, (_, row) in enumerate(slowest.iterrows(), 1):
                logging.info(f"{idx}. {row['file_name']} | {row['total_time_s']} s | nodes={row['num_nodes']} | edges={row['num_edges']}")
        
        logging.info("="*80 + "\n")


def normalize_geo_features(geo_data):
    """Normalize lon, lat, and layer values for model input."""
    scaler = StandardScaler()
    normalized = scaler.fit_transform(geo_data)
    return normalized, scaler

def calculate_pan_direction(curr_lon, next_lon, curr_lat, next_lat):
    """Calculate pan direction: 1=east, 2=south, 3=west, 4=north, 0=no movement."""
    delta_lon = next_lon - curr_lon
    delta_lat = next_lat - curr_lat
    
    if abs(delta_lon) < 1e-8 and abs(delta_lat) < 1e-8:
        return 0
    
    if abs(delta_lon) > abs(delta_lat):
        return 1 if delta_lon > 0 else 3
    else:
        return 4 if delta_lat > 0 else 2

def calculate_operation_features(df_sorted):
    """Calculate operation features: zoom type, zoom layer, pan distance, pan direction, and operation distance."""
    n = len(df_sorted)
    operation_features = np.zeros((n, 5))
    window_size = 3
    
    for i in range(n-1):
        curr_layer = df_sorted.iloc[i]['layer']
        next_layer = df_sorted.iloc[i+1]['layer']
        curr_col = df_sorted.iloc[i]['col']
        next_col = df_sorted.iloc[i+1]['col']
        curr_row = df_sorted.iloc[i]['row']
        next_row = df_sorted.iloc[i+1]['row']
        curr_lon = df_sorted.iloc[i]['lon']
        next_lon = df_sorted.iloc[i+1]['lon']
        curr_lat = df_sorted.iloc[i]['lat']
        next_lat = df_sorted.iloc[i+1]['lat']
        
        if curr_layer != next_layer:
            zoom_type = 1 if next_layer > curr_layer else 2
            zoom_layer = abs(next_layer - curr_layer)
            pan_distance = 0
            pan_direction = 0
            operation_distance = zoom_layer
        else:
            zoom_type = 0
            zoom_layer = 0
            pan_distance = abs(next_col - curr_col) + abs(next_row - curr_row)
            pan_direction = calculate_pan_direction(curr_lon, next_lon, curr_lat, next_lat)
            operation_distance = pan_distance / window_size
        
        operation_features[i] = [zoom_type, zoom_layer, pan_distance, pan_direction, operation_distance]
    
    return operation_features

def calculate_time_features(df_sorted):
    """Calculate dwell time as the time difference between adjacent trajectory points."""
    n = len(df_sorted)
    time_features = np.zeros(n)
    df_sorted['time'] = pd.to_datetime(df_sorted['time'])
    
    for i in range(n-1):
        time_diff = (df_sorted.iloc[i+1]['time'] - df_sorted.iloc[i]['time']).total_seconds()
        time_features[i] = time_diff
    
    return time_features

def compute_knn_density_with_levels(node_coords, node_levels, k=10):
    n_nodes = len(node_coords)
    all_neighbors = []
    unique_levels = np.unique(node_levels)
    min_level = unique_levels.min()
    max_level = unique_levels.max()
    
    if n_nodes <= 3:
        logging.warning("Too few nodes for level-aware density; falling back to nearest-neighbor density.")
        k = max(1, n_nodes - 1)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(node_coords)
        distances, _ = nbrs.kneighbors(node_coords)
        mean_dist = distances.mean(axis=1)
        density = 1 / (mean_dist + 1e-8)
        return StandardScaler().fit_transform(density.reshape(-1, 1))
    
    for i in range(n_nodes):
        current_level = node_levels[i]
        target_levels = {current_level}
        if current_level > min_level:
            target_levels.add(current_level - 1)
        if current_level < max_level:
            target_levels.add(current_level + 1)
        
        level_mask = np.isin(node_levels, list(target_levels))
        level_mask[i] = False
        valid_indices = np.where(level_mask)[0]
        valid_coords = node_coords[valid_indices]
        
        if len(valid_indices) < k:
            other_levels_mask = ~np.isin(node_levels, list(target_levels))
            other_levels_mask[i] = False
            other_indices = np.where(other_levels_mask)[0]
            combined_indices = np.concatenate([valid_indices, other_indices])
            combined_coords = node_coords[combined_indices]
            
            current_coord = node_coords[i].reshape(1, -1)
            dists = np.linalg.norm(combined_coords - current_coord, axis=1)
            sorted_indices = np.argsort(dists)
            nearest_indices = combined_indices[sorted_indices[:k]]
            all_neighbors.append(nearest_indices)
        else:
            current_coord = node_coords[i].reshape(1, -1)
            dists = np.linalg.norm(valid_coords - current_coord, axis=1)
            sorted_indices = np.argsort(dists)
            nearest_indices = valid_indices[sorted_indices[:k]]
            all_neighbors.append(nearest_indices)
    
    mean_distances = []
    for i, neighbors in enumerate(all_neighbors):
        if len(neighbors) == 0:
            mean_distances.append(1e-8)
            continue
        distances = np.linalg.norm(node_coords[neighbors] - node_coords[i], axis=1)
        mean_dist = np.mean(distances)
        mean_distances.append(mean_dist)
    
    mean_distances = np.array(mean_distances)
    density = 1 / (mean_distances + 1e-8)
    scaler = StandardScaler()
    density_scaled = scaler.fit_transform(density.reshape(-1, 1))
    return density_scaled

def planar_opdistance(node1_rc, node2_rc, samelayer_penalty=1.0, window=3):
    row1, col1, layer1 = node1_rc
    row2, col2, layer2 = node2_rc
    delta_z = abs(layer1 - layer2)
    
    if layer1 < layer2:
        base_row, base_col = row1, col1
        target_row, target_col = row2, col2
    else:
        base_row, base_col = row2, col2
        target_row, target_col = row1, col1
    
    scaled_row = base_row * (2 ** delta_z)
    scaled_col = base_col * (2 ** delta_z)
    tiledis = abs(scaled_row - target_row) + abs(scaled_col - target_col)
    return (tiledis / window) * samelayer_penalty

def find_adjacent_layer_nodes(node_id, node_layer, all_nodes):
    current_layer = node_layer[node_id]
    for delta in range(1, 5):
        upper_layer = current_layer + delta
        upper_nodes = [n for n in all_nodes if node_layer[n] == upper_layer]
        if upper_nodes:
            return upper_nodes
        
        lower_layer = current_layer - delta
        lower_nodes = [n for n in all_nodes if node_layer[n] == lower_layer]
        if lower_nodes:
            return lower_nodes
    return []

def force_connect_components(G, node_row_col, node_time, samelayer_penalty=1.0, window=3):
    iteration = 0
    max_iterations = 100
    total_added_edges = 0
    
    while True:
        iteration += 1
        if iteration > max_iterations:
            logging.warning("Maximum component-connection iterations reached; graph may remain disconnected.")
            break
            
        components = list(nx.connected_components(G))
        num_components = len(components)
        if num_components == 1:
            logging.info(f"Graph connected after adding {total_added_edges} forced edges.")
            return G
        
        logging.info(f"Connecting {num_components} graph components.")
        candidate_edges = []
        for i in range(num_components):
            for j in range(i + 1, num_components):
                for node1 in components[i]:
                    for node2 in components[j]:
                        dist = planar_opdistance(node_row_col[node1], node_row_col[node2], samelayer_penalty, window)
                        candidate_edges.append((node1, node2, dist))
        
        candidate_edges.sort(key=lambda x: x[2])
        added_edges = 0
        for node1, node2, dist in candidate_edges:
            if not nx.has_path(G, node1, node2):
                G.add_edge(node1, node2, weight=dist, time_diff=-1, edge_type='forced')
                added_edges += 1
                total_added_edges += 1
                if added_edges % 10 == 0 and len(list(nx.connected_components(G))) == 1:
                    break
        
        if added_edges == 0:
            logging.warning("No valid candidate edges were found for component connection.")
            break
    
    return G

def build_graph(data, threshold, time_threshold=10, samelayer_penalty=1.0, window=3, 
                max_same_layer=8, max_upper_layer=4, max_lower_layer=4,
                base_degree_threshold=16, layer_degree_bonus=2):
    G = nx.Graph()
    node_row_col = {}
    node_time = {}
    node_layer = {}
    node_lon_lat = {}
    node_label = {}
    node_label_int = {}
    label_column = 'MSPGL_label' if 'MSPGL_label' in data.columns else 'label'
    
    for index, row in data.iterrows():
        node_id = row['ID']
        lon = row['lon']
        lat = row['lat']
        layer = row['layer']
        time = row['time']
        label_raw = str(row[label_column]).strip().upper()
        row_csv = row['row']
        col_csv = row['col']
        
        if label_raw == 'Y':
            label_int = 1
        elif label_raw == 'N':
            label_int = 0
        else:
            label_int = 0
            logging.warning(f"Unexpected label '{label_raw}' encountered; it is treated as non-target.")
        
        G.add_node(node_id)
        node_row_col[node_id] = (row_csv, col_csv, layer)
        node_time[node_id] = time
        node_layer[node_id] = layer
        node_lon_lat[node_id] = (lon, lat)
        node_label[node_id] = label_raw
        node_label_int[node_id] = label_int
    
    num_nodes = len(G.nodes())
    nodes = list(G.nodes())
    logging.debug("MSPGL debug message")
    
    for i, node1 in enumerate(nodes):
        node1_rc = node_row_col[node1]
        layer1 = node_layer[node1]
        
        same_layer_neighbors = []
        for j, node2 in enumerate(nodes):
            if i == j:
                continue
            if node_layer[node2] != layer1:
                continue
            
            time_diff = abs(node_time[node1] - node_time[node2])
            if time_diff >= time_threshold:
                continue
            dist = planar_opdistance(node1_rc, node_row_col[node2], samelayer_penalty, window)
            if dist < threshold:
                same_layer_neighbors.append((node2, dist, time_diff))
        
        same_layer_neighbors.sort(key=lambda x: x[1])
        for neighbor, dist, time_diff in same_layer_neighbors[:max_same_layer]:
            G.add_edge(node1, neighbor, weight=dist, time_diff=time_diff, edge_type='same')
        
        adjacent_nodes = find_adjacent_layer_nodes(node1, node_layer, nodes)
        if adjacent_nodes:
            upper_neighbors = []
            lower_neighbors = []
            
            for node2 in adjacent_nodes:
                node2_rc = node_row_col[node2]
                layer2 = node_layer[node2]
                time_diff = abs(node_time[node1] - node_time[node2])
                if time_diff >= time_threshold:
                    continue
                dist = planar_opdistance(node1_rc, node2_rc, samelayer_penalty, window)
                
                if layer2 > layer1:
                    upper_neighbors.append((node2, dist, time_diff))
                elif layer2 < layer1:
                    lower_neighbors.append((node2, dist, time_diff))
            
            upper_neighbors.sort(key=lambda x: x[1])
            lower_neighbors.sort(key=lambda x: x[1])
            for neighbor, dist, time_diff in upper_neighbors[:max_upper_layer]:
                G.add_edge(node1, neighbor, weight=dist, time_diff=time_diff, edge_type='upper')
            for neighbor, dist, time_diff in lower_neighbors[:max_lower_layer]:
                G.add_edge(node1, neighbor, weight=dist, time_diff=time_diff, edge_type='lower')
    
    isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
    for node in isolated_nodes:
        min_dist = float('inf')
        nearest_node = None
        for other_node in G.nodes():
            if node == other_node:
                continue
            time_diff = abs(node_time[node] - node_time[other_node])
            if time_diff >= time_threshold:
                continue
            dist = planar_opdistance(node_row_col[node], node_row_col[other_node], samelayer_penalty, window)
            if dist < min_dist:
                min_dist = dist
                nearest_node = other_node
        
        if nearest_node and min_dist < threshold:
            G.add_edge(node, nearest_node, weight=min_dist, time_diff=time_diff, edge_type='isolated')
    
    remaining_isolated = [node for node in G.nodes() if G.degree(node) == 0]
    if remaining_isolated:
        logging.info(f"Connecting {len(remaining_isolated)} remaining isolated nodes to their nearest neighbors.")
        for node in remaining_isolated:
            min_dist = float('inf')
            nearest_node = None
            for other_node in G.nodes():
                if node == other_node:
                    continue
                dist = planar_opdistance(node_row_col[node], node_row_col[other_node], samelayer_penalty, window)
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = other_node
            
            if nearest_node:
                G.add_edge(node, nearest_node, weight=min_dist, time_diff=-1, edge_type='forced')
                logging.debug("MSPGL debug message")
    
    over_degree_count = 0
    for node in list(G.nodes()):
        current_layer = node_layer[node]
        dynamic_threshold = base_degree_threshold + (current_layer * layer_degree_bonus)
        current_degree = G.degree(node)
        
        if current_degree <= dynamic_threshold:
            continue
        
        over_degree_count += 1
        edges = []
        for neighbor in G.neighbors(node):
            edge_data = G[node][neighbor]
            edge_type = edge_data['edge_type']
            dist = edge_data['weight']
            neighbor_layer = node_layer[neighbor]
            layer_diff = abs(neighbor_layer - current_layer)
            
            if edge_type == 'forced':
                priority = 1
            elif layer_diff == 1:
                priority = 2
            elif layer_diff > 1:
                priority = 3
            else:
                priority = 4
            
            edges.append((neighbor, dist, priority, edge_type))
        
        edges.sort(key=lambda x: (x[2], x[1]))
        keep_edges = set(neighbor for neighbor, _, _, _ in edges[:dynamic_threshold])
        removed_edges = 0
        
        for neighbor in list(G.neighbors(node)):
            if neighbor not in keep_edges:
                if nx.has_path(G, node, neighbor, cutoff=2):
                    G.remove_edge(node, neighbor)
                    removed_edges += 1
        
        logging.debug("MSPGL debug message")
    
    if over_degree_count > 0:
        logging.info(f"Degree pruning was applied to {over_degree_count} nodes.")
    
    if not nx.is_connected(G):
        G = force_connect_components(G, node_row_col, node_time, samelayer_penalty, window)
    
    return (G, node_row_col, node_time, node_layer, node_lon_lat, node_label, node_label_int)

# ---------------------- 4. Graph visualization ----------------------
def visualize_graph(G, node_lon_lat, node_layer, node_label, output_dir, filename, subfolder_name):
    nodes = list(G.nodes())
    x_values = [node_lon_lat[node][0] for node in nodes]
    y_values = [node_lon_lat[node][1] for node in nodes]
    z_values = [node_layer[node] for node in nodes]
    node_labels = [node_label[node] for node in nodes]
    components = list(nx.connected_components(G))
    component_colors = [0] * len(nodes)
    for i, comp in enumerate(components):
        for idx, node in enumerate(nodes):
            if node in comp:
                component_colors[idx] = i
    
    node_text = [f'ID: {node}<br>Lon: {x:.4f}, Lat: {y:.4f}<br>Layer: {z}<br>Label: {l}<br>Degree: {G.degree(node)}'
                 for node, x, y, z, l in zip(nodes, x_values, y_values, z_values, node_labels)]
    
    edge_x, edge_y, edge_z = [], [], []
    edge_text, edge_colors = [], []
    for u, v, data in G.edges(data=True):
        u_lon, u_lat = node_lon_lat[u]
        v_lon, v_lat = node_lon_lat[v]
        u_layer, v_layer = node_layer[u], node_layer[v]
        
        edge_x.extend([u_lon, v_lon, None])
        edge_y.extend([u_lat, v_lat, None])
        edge_z.extend([u_layer, v_layer, None])
        
        dist = round(data['weight'], 4)
        time_diff = data['time_diff']
        edge_type = data['edge_type']
        
        edge_color_map = {'same': 'red', 'upper': 'blue', 'lower': 'green', 'isolated': 'orange', 'forced': 'purple'}
        edge_color = edge_color_map.get(edge_type, 'gray')
        
        edge_colors.append(edge_color)
        edge_text.append(f'Type: {edge_type}<br>Distance: {dist}<br>Time Diff: {time_diff}')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines', line=dict(color=edge_colors, width=1),
        hovertext=edge_text, hoverinfo='text'
    ))
    fig.add_trace(go.Scatter3d(
        x=x_values, y=y_values, z=z_values,
        mode='markers', marker=dict(size=4, color=component_colors, colorscale='Viridis', opacity=0.8),
        hovertext=node_text, hoverinfo='text'
    ))
    
    connectivity_info = "globally connected" if nx.is_connected(G) else f"{len(components)} connected components"
    fig.update_layout(
        title=f"3D Graph Structure - {os.path.basename(filename)} ({connectivity_info})",
        scene=dict(xaxis_title='Longitude', yaxis_title='Latitude', zaxis_title='Layer'),
        hovermode='closest',
        height=800
    )
    
    subfolder_output_dir = os.path.join(output_dir, subfolder_name)
    Path(subfolder_output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(subfolder_output_dir, f'{base_name}_graph_vis_{timestamp}.html')
    fig.write_html(save_path)
    logging.info(f"Graph visualization saved to: {save_path}")
    return save_path

def convert_nx_to_pyg(G, node_row_col, node_time, node_layer, node_lon_lat, node_label, node_label_int,
                      normalized_geo_features, operation_features, time_features):
    """
    Build PyG node features without the leaves field.
    Feature layout before density append: ID, normalized lon/lat/layer, operation features, and time interval.
    """
    nodes = sorted(list(G.nodes()))
    node_id_to_idx = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    
    # 1. Build node features without leaves.
    x = []
    for idx, node in enumerate(nodes):
        id_embed = idx
        norm_lon, norm_lat, norm_layer = normalized_geo_features[idx]
        op_feat = operation_features[idx]
        time_feat = time_features[idx]
        base_feat = [id_embed, norm_lon, norm_lat, norm_layer] + op_feat.tolist() + [time_feat]
        x.append(base_feat)
    x = torch.tensor(x, dtype=torch.float)
    
    y = torch.tensor([node_label_int[node] for node in nodes], dtype=torch.long)
    
    edge_index = []
    edge_attr = []
    for u, v, data in G.edges(data=True):
        u_idx = node_id_to_idx[u]
        v_idx = node_id_to_idx[v]
        
        edge_index.append([u_idx, v_idx])
        edge_index.append([v_idx, u_idx])
        
        dist = data['weight']
        time_diff = data['time_diff']
        edge_type = data['edge_type']
        
        type_encoding = [0]*5
        type_map = {'same':0, 'upper':1, 'lower':2, 'isolated':3, 'forced':4}
        if edge_type in type_map:
            type_encoding[type_map[edge_type]] = 1
        
        edge_feat = [dist, time_diff] + type_encoding
        edge_attr.append(edge_feat)
        edge_attr.append(edge_feat)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    layer_list = [node_layer[node] for node in nodes]
    layer_tensor = torch.tensor(layer_list, dtype=torch.long)
    
    data = Data(
        x=x, 
        y=y, 
        edge_index=edge_index, 
        edge_attr=edge_attr,
        layer=layer_tensor,
        num_nodes=num_nodes, 
        node_id_to_idx=node_id_to_idx,
        node_label_raw=node_label,
        node_lon_lat=node_lon_lat,
        node_layer=node_layer
    )
    return data

def process_and_save_files(input_dir, output_vis_dir, output_pyg_dir, 
                          threshold=1500, time_threshold=3, knn_k=10, **kwargs):
    tracker = PerformanceTracker()
    tracker.log_device_info()
    
    Path(output_vis_dir).mkdir(parents=True, exist_ok=True)
    Path(output_pyg_dir).mkdir(parents=True, exist_ok=True)
    
    failed_files = {}
    failed_by_reason = {}
    
    subfolders = []
    for entry in os.scandir(input_dir):
        if entry.is_dir():
            subfolders.append(entry.name)
    
    if not subfolders:
        logging.warning("No subfolders were found; CSV files will be read directly from the input directory.")
        csv_files = [(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
        subfolder_name = "root"
    else:
        logging.info(f"All {success_count} files were processed successfully.")
        csv_files = []
        for subfolder in subfolders:
            subfolder_path = os.path.join(input_dir, subfolder)
            subfolder_csvs = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
            if not subfolder_csvs:
                logging.warning(f"No CSV files found in subfolder: {subfolder}")
                continue
            for csv in subfolder_csvs:
                csv_files.append((subfolder_path, subfolder, csv))
    
    if not csv_files:
        logging.error("No CSV files were found for graph preprocessing.")
        return
    
    logging.info(f"Preparing to process {len(csv_files)} CSV files.")
    success_count = 0
    
    for item in csv_files:
        file_start_time = time.time()
        stages_time = {}
        
        if len(item) == 2:
            subfolder_path, csv_filename = item
            subfolder_name = "root"
        else:
            subfolder_path, subfolder_name, csv_filename = item
        
        file_path = os.path.join(subfolder_path, csv_filename)
        file_key = f"{subfolder_name}/{csv_filename}"
        logging.info(f"Processing file: {file_key}")
        
        try:
            stage_start = time.time()
            try:
                data = pd.read_csv(file_path)
            except Exception as e:
                reason = "Processing error"
                failed_files[file_key] = {
                    'reason': reason,
                    'error': str(e),
                    'subfolder': subfolder_name
                }
                failed_by_reason.setdefault(reason, []).append(file_key)
                logging.error(f"MSPGL processing error: {str(e)}")
                tracker.add_result(file_key, 'failed', time.time() - file_start_time, subfolder_name, error_msg=str(e))
                continue
            
            stages_time['csv_read'] = time.time() - stage_start
            
            required_cols = ['ID', 'lon', 'lat', 'layer', 'time', 'row', 'col']
            missing_cols = set(required_cols) - set(data.columns)
            label_column = 'MSPGL_label' if 'MSPGL_label' in data.columns else 'label'
            if label_column not in data.columns:
                missing_cols.add('MSPGL_label')
            if missing_cols:
                reason = "Processing error"
                error_msg = "Processing error"
                failed_files[file_key] = {
                    'reason': reason,
                    'error': error_msg,
                    'subfolder': subfolder_name
                }
                failed_by_reason.setdefault(reason, []).append(file_key)
                logging.error(f"MSPGL processing error: missing required columns {sorted(missing_cols)}")
                tracker.add_result(file_key, 'failed', time.time() - file_start_time, subfolder_name, error_msg=error_msg)
                continue
            
            if len(data) == 0:
                reason = "Processing error"
                error_msg = "Processing error"
                failed_files[file_key] = {
                    'reason': reason,
                    'error': error_msg,
                    'subfolder': subfolder_name
                }
                failed_by_reason.setdefault(reason, []).append(file_key)
                logging.error(f"MSPGL processing error: {str(e)}")
                tracker.add_result(file_key, 'failed', time.time() - file_start_time, subfolder_name, error_msg=error_msg)
                continue
            
            data_sorted = data.sort_values(['ID', 'time']).reset_index(drop=True)
            
            stage_start = time.time()
            try:
                geo_data = data_sorted[['lon', 'lat', 'layer']].values
                normalized_geo, _ = normalize_geo_features(geo_data)
                operation_features = calculate_operation_features(data_sorted)
                time_features = calculate_time_features(data_sorted)
            except Exception as e:
                reason = "Processing error"
                failed_files[file_key] = {
                    'reason': reason,
                    'error': str(e),
                    'subfolder': subfolder_name
                }
                failed_by_reason.setdefault(reason, []).append(file_key)
                logging.error(f"MSPGL processing error: {str(e)}")
                tracker.add_result(file_key, 'failed', time.time() - file_start_time, subfolder_name, error_msg=str(e))
                continue
            
            stages_time['feature_calc'] = time.time() - stage_start
            
            stage_start = time.time()
            try:
                G, node_row_col, node_time, node_layer, node_lon_lat, node_label, node_label_int = build_graph(
                    data, threshold, time_threshold, **kwargs
                )
                
                if len(G.nodes()) == 0:
                    raise ValueError("MSPGL processing error")
                if len(G.edges()) == 0:
                    raise ValueError("MSPGL processing error")
                    
            except Exception as e:
                reason = "Processing error"
                failed_files[file_key] = {
                    'reason': reason,
                    'error': str(e),
                    'subfolder': subfolder_name
                }
                failed_by_reason.setdefault(reason, []).append(file_key)
                logging.error(f"MSPGL processing error: {str(e)}")
                tracker.add_result(file_key, 'failed', time.time() - file_start_time, subfolder_name, error_msg=str(e))
                continue
            
            stages_time['graph_build'] = time.time() - stage_start
            
            stage_start = time.time()
            try:
                visualize_graph(G, node_lon_lat, node_layer, node_label, output_vis_dir, file_path, subfolder_name)
            except Exception as e:
                reason = "Processing error"
                failed_files[file_key] = {
                    'reason': reason,
                    'error': str(e),
                    'subfolder': subfolder_name
                }
                failed_by_reason.setdefault(reason, []).append(file_key)
                logging.error(f"MSPGL processing error: {str(e)}")
                tracker.add_result(file_key, 'failed', time.time() - file_start_time, subfolder_name, error_msg=str(e))
                continue
            
            stages_time['visualization'] = time.time() - stage_start
            
            stage_start = time.time()
            try:
                pyg_data = convert_nx_to_pyg(
                    G, node_row_col, node_time, node_layer, node_lon_lat, node_label, node_label_int,
                    normalized_geo, operation_features, time_features
                )
            except Exception as e:
                reason = "Processing error"
                failed_files[file_key] = {
                    'reason': reason,
                    'error': str(e),
                    'subfolder': subfolder_name
                }
                failed_by_reason.setdefault(reason, []).append(file_key)
                logging.error(f"MSPGL processing error: {str(e)}")
                tracker.add_result(file_key, 'failed', time.time() - file_start_time, subfolder_name, error_msg=str(e))
                continue
            
            stages_time['pyg_convert'] = time.time() - stage_start
            
            stage_start = time.time()
            try:
                nodes = sorted(G.nodes())
                node_coords = np.array([node_lon_lat[node] for node in nodes])
                node_levels = np.array([node_layer[node] for node in nodes])
                density_features = compute_knn_density_with_levels(node_coords, node_levels, k=knn_k)
                density_tensor = torch.tensor(density_features, dtype=torch.float)
                pyg_data.x = torch.cat([pyg_data.x, density_tensor], dim=1)
            except Exception as e:
                reason = "Processing error"
                failed_files[file_key] = {
                    'reason': reason,
                    'error': str(e),
                    'subfolder': subfolder_name
                }
                failed_by_reason.setdefault(reason, []).append(file_key)
                logging.error(f"MSPGL processing error: {str(e)}")
                tracker.add_result(file_key, 'failed', time.time() - file_start_time, subfolder_name, error_msg=str(e))
                continue
            
            stages_time['density_calc'] = time.time() - stage_start
            
            stage_start = time.time()
            try:
                subfolder_pyg_dir = os.path.join(output_pyg_dir, subfolder_name)
                Path(subfolder_pyg_dir).mkdir(parents=True, exist_ok=True)
                
                base_name = os.path.splitext(csv_filename)[0]
                pyg_save_path = os.path.join(subfolder_pyg_dir, f'{base_name}_gnn_input.pt')
                torch.save(pyg_data, pyg_save_path)
            except Exception as e:
                reason = "Processing error"
                failed_files[file_key] = {
                    'reason': reason,
                    'error': str(e),
                    'subfolder': subfolder_name
                }
                failed_by_reason.setdefault(reason, []).append(file_key)
                logging.error(f"MSPGL processing error: {str(e)}")
                tracker.add_result(file_key, 'failed', time.time() - file_start_time, subfolder_name, error_msg=str(e))
                continue
            
            stages_time['save'] = time.time() - stage_start
            
            nodes_list = sorted(G.nodes())
            label_counts = pd.Series([node_label_int[node] for node in nodes_list]).value_counts()
            
            total_time = time.time() - file_start_time
            
            logging.info(f"Finished {file_key} in {total_time:.2f} seconds.")
            logging.info(f"  - Nodes: {pyg_data.num_nodes}")
            logging.info(f"  - Edges: {pyg_data.edge_index.size(1) // 2}")
            logging.info(f"  - Node feature dimension: {pyg_data.x.size(1)} (ID + normalized geo + operation + time + density)")
            logging.info(f"  - PyG graph saved to: {pyg_path}")
            logging.info(f"  - Visualization saved to: {vis_path}")
            tracker.add_result(
                file_key,
                'success',
                total_time,
                subfolder_name,
                num_nodes=pyg_data.num_nodes,
                num_edges=pyg_data.edge_index.size(1) // 2,
                feature_dim=pyg_data.x.size(1),
                stages_time=stages_time
            )
            
            success_count += 1
        
        except Exception as e:
            reason = "Processing error"
            failed_files[file_key] = {
                'reason': reason,
                'error': str(e),
                'subfolder': subfolder_name
            }
            failed_by_reason.setdefault(reason, []).append(file_key)
            logging.error(f"MSPGL processing error: {str(e)}", exc_info=True)
            tracker.add_result(file_key, 'failed', time.time() - file_start_time, subfolder_name, error_msg=str(e))
    
    failure_count = len(failed_files)
    
    logging.info("\n" + "="*80)
    logging.info("Batch Processing Summary")
    logging.info("="*80)
    
    if failure_count > 0:
        logging.info(f"Failed files: {failure_count}")
        
        for reason, files in sorted(failed_by_reason.items(), key=lambda x: -len(x[1])):
            logging.info(f"Failure reason: {reason} ({len(files)} files)")
            for i, file_key in enumerate(sorted(files), 1):
                file_info = failed_files[file_key]
                error_detail = file_info['error']
                if len(error_detail) > 100:
                    error_detail = error_detail[:97] + "..."
                logging.info(f"{i:2d}. {file_key}")
                logging.info(f"    Error: {error_detail}")
            logging.info(f"└─")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            failed_list_path = os.path.join(output_pyg_dir, f'FAILED_FILES_{timestamp}.txt')
            
            with open(failed_list_path, 'w', encoding='utf-8') as f:
                f.write("MSPGL processing report\n")
                f.write("MSPGL processing report\n")
                f.write("MSPGL processing report\n")
                
                for reason, files in sorted(failed_by_reason.items(), key=lambda x: -len(x[1])):
                    f.write("MSPGL processing report\n")
                    f.write("-" * 80 + "\n")
                    for i, file_key in enumerate(sorted(files), 1):
                        file_info = failed_files[file_key]
                        f.write(f"{i:3d}. {file_key}\n")
                        f.write("MSPGL processing report\n")
            
            logging.info(f"Failure report saved to: {failure_report_path}")
            logging.info(f"Successful files: {success_count}")
        except Exception as e:
            logging.error(f"MSPGL processing error: {str(e)}")
    else:
        logging.info(f"All {success_count} files were processed successfully.")
    
    logging.info("="*80 + "\n")
    
    tracker.save_results_to_csv(output_pyg_dir)


if __name__ == "__main__":
    INPUT_DIR = "./data/raw_csv"
    OUTPUT_VIS_DIR = "./outputs/graph_visualization"
    OUTPUT_PYG_DIR = "./data/pyg_graphs"
    
    GRAPH_PARAMS = {
        'threshold': 50.33,
        'time_threshold': 3,
        'samelayer_penalty': 1.0,
        'window': 3,
        'max_same_layer': 8,
        'max_upper_layer': 4,
        'max_lower_layer': 4,
        'base_degree_threshold': 16,
        'layer_degree_bonus': 2
    }
    
    KNN_K = 10
    
    process_and_save_files(
        input_dir=INPUT_DIR,
        output_vis_dir=OUTPUT_VIS_DIR,
        output_pyg_dir=OUTPUT_PYG_DIR,
        knn_k=KNN_K,
        **GRAPH_PARAMS
    )

# %% [markdown] Cell 3
# # Training the MSPGL model

# %% Cell 4
# Force CUDA synchronization for easier GPU error tracing.
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.data import Data
from torch.nn import Dropout
from torch_geometric.nn import MessagePassing
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                             confusion_matrix, classification_report,
                             precision_recall_curve, roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import random
import joblib

# ------------------------
# Core configuration for the two-stage layer-aware model.
# ------------------------
data_root = "./data/pyg_graphs"
save_dir  = "./outputs/model"
os.makedirs(save_dir, exist_ok=True)

# Dataset split ratio
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Base hyperparameters
batch_size = 1        
lr = 5e-4              
weight_decay = 5e-5    
num_epochs = 200       
hidden_dim = 128       
dropout_rate = 0.2     
patience = 20          
PHASE1_TARGET_RECALL = 0.8
PHASE2_THRESHOLD = 0.5
layer_emb_dim = 32
max_layer = 0

# ------------------------
# Neighborhood consistency features used by Stage 2.
# ------------------------
def compute_veto_features(node_id, edge_index, phase1_prob):
    src, dst = edge_index
    nbr_mask_src = (src == node_id)
    nbr_mask_dst = (dst == node_id)
    nbr_ids_src = dst[nbr_mask_src]
    nbr_ids_dst = src[nbr_mask_dst]
    nbr_ids = torch.cat([nbr_ids_src, nbr_ids_dst]).unique()

    if nbr_ids.numel() == 0:
        return [0.0, 0.0, 0.0]

    nbr_probs = phase1_prob[nbr_ids]
    nbr_pos_ratio = (nbr_probs > 0.5).float().mean().item()
    prob_self = phase1_prob[node_id].item()
    prob_minus_nbr_mean = prob_self - nbr_probs.mean().item()
    prob_minus_nbr_max = prob_self - nbr_probs.max().item()

    return [nbr_pos_ratio, prob_minus_nbr_mean, prob_minus_nbr_max]

# ------------------------
# 1. Data loading and random split
# ------------------------
def clean_data(data):
    core_keys = ['x', 'y', 'edge_index', 'edge_attr', 'num_nodes', 'layer']
    new_data = Data()
    for key in core_keys:
        if hasattr(data, key):
            setattr(new_data, key, getattr(data, key))
    if hasattr(data, 'x'):
        if data.x.shape[1] == 12:
            # Drop the legacy leaves column from previously generated PyG files.
            data.x = torch.cat([data.x[:, :10], data.x[:, 11:]], dim=1)
        new_data.x = data.x
        new_data.num_nodes = data.x.shape[0]
    new_data.node_id = torch.arange(new_data.num_nodes)
    y_count = (new_data.y == 1).sum().item() if hasattr(new_data, 'y') else 0
    y_ratio = y_count / max(new_data.num_nodes, 1)
    new_data.y_count = y_count
    new_data.y_ratio = y_ratio
    return new_data

def load_all_data(data_root):
    global max_layer
    all_data = []
    skipped_files = []
    all_layers = []
    
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.endswith(".pt"):
                try:
                    data = torch.load(os.path.join(root, f), map_location="cpu")
                    if not isinstance(data, Data):
                        skipped_files.append(f)
                        continue
                    
                    data = clean_data(data)
                    if data.num_nodes == 0:
                        skipped_files.append(f)
                        continue
                    
                    if hasattr(data, 'layer'):
                        all_layers.extend(data.layer.cpu().numpy())
                    
                    all_data.append(data)
                    
                except Exception as e:
                    print(f"Skipping {f}: {e}")
                    skipped_files.append(f)
                    continue
    
    if all_layers:
        max_layer = int(np.max(all_layers))
    else:
        max_layer = 10
    print(f"Loaded {len(all_data)} PyG graph files from {root_folder}.")
    print(f"Skipped files: {len(skipped_files)}")
    print(f"Maximum level detected: {max_layer}")
    print(f"Stage-1 input excludes the leaves feature; input dimension is taken from each PyG graph.")
    
    return all_data

def split_dataset(all_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    # English reviewer note: message translated from original processing output.
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    random.shuffle(all_data)
    n_total = len(all_data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train+n_val]
    test_data = all_data[n_train+n_val:]
    
    def get_label_stats(dataset):
        all_labels = []
        for data in dataset:
            all_labels.extend(data.y.cpu().numpy())
        return Counter(all_labels)
    
    train_stats = get_label_stats(train_data)
    val_stats = get_label_stats(val_data)
    test_stats = get_label_stats(test_data)
    
    print(f"Dataset split sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    print(f"Train label counts: {dict(train_stats)}")
    print(f"Validation label counts: {dict(val_stats)}")
    print(f"Test label counts: {dict(test_stats)}")
    
    def get_imbalance_ratio(label_counter):
        if 0 not in label_counter:
            label_counter[0] = 1
        if 1 not in label_counter:
            label_counter[1] = 1
        minority = 1 if label_counter[1] < label_counter[0] else 0
        majority = 1 - minority
        return label_counter[majority] / label_counter[minority]
    
    train_imbalance = get_imbalance_ratio(train_stats)
    val_imbalance = get_imbalance_ratio(val_stats)
    test_imbalance = get_imbalance_ratio(test_stats)
    
    print(f"Train imbalance ratio: {train_imbalance:.2f}")
    print(f"Validation imbalance ratio: {val_imbalance:.2f}")
    print(f"Test imbalance ratio: {test_imbalance:.2f}")
    print("Dataset split completed.")
    
    return train_data, val_data, test_data, train_stats

# ------------------------
# 2. Balanced sampling
# ------------------------
def get_balanced_sampler(dataset):
    sample_weights = []
    BASE_SAMPLE_WEIGHT = 1.0
    Y_COUNT_COEFF = 12
    Y_RATIO_COEFF = 8
    
    for data in dataset:
        y_count = data.y_count if hasattr(data, 'y_count') else 0
        y_ratio = data.y_ratio if hasattr(data, 'y_ratio') else 0.0
        
        weight = (y_count * Y_COUNT_COEFF) + (y_ratio * Y_RATIO_COEFF) + BASE_SAMPLE_WEIGHT
        sample_weights.append(weight)
    
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    sample_weights = sample_weights / sample_weights.max() * 10
    
    print(f"Training label counts for sampler: {dict(label_counts)}")
    print(f"Class weights: {class_weights}")
    print("Weighted sampler created for class-imbalanced node labels.")
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    return sampler

# ------------------------
# 3. Model definition
# ------------------------
class EdgeTypeAwareGATConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        dropout=0.0,
        edge_type_dim=5,
        negative_slope=0.2,
        concat=True,
        bias=True,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.edge_type_bias = nn.Parameter(torch.Tensor(edge_type_dim, heads))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.edge_type_bias)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        H, C = self.heads, self.out_channels
        x = self.lin(x).view(-1, H, C)
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        alpha = ((x_i * self.att_dst).sum(dim=-1) + (x_j * self.att_src).sum(dim=-1))
        edge_type_onehot = edge_attr[:, 2:]
        edge_bias = torch.matmul(edge_type_onehot, self.edge_type_bias)
        alpha = alpha + edge_bias
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

class EdgeTypeAwareGAT(nn.Module):
    def __init__(
        self,
        input_dim,
        layer_emb_dim,
        max_layer,
        hidden_dim=128,
        num_heads=4,
        num_classes=2,
        dropout=0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = num_heads
        self.layer_emb_dim = layer_emb_dim
        
        self.layer_embedding = nn.Embedding(max_layer + 1, layer_emb_dim)
        nn.init.xavier_uniform_(self.layer_embedding.weight)

        self.input_proj = nn.Linear(input_dim + layer_emb_dim, hidden_dim * num_heads)

        self.gat1 = EdgeTypeAwareGATConv(
            hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, edge_type_dim=5
        )
        self.bn1 = BatchNorm(hidden_dim * num_heads)

        self.gat2 = EdgeTypeAwareGATConv(
            hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, edge_type_dim=5
        )
        self.bn2 = BatchNorm(hidden_dim * num_heads)

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * num_heads, num_classes)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        layer = data.layer

        layer_emb = self.layer_embedding(layer)
        x_combined = torch.cat([x, layer_emb], dim=1)
        x_proj = self.input_proj(x_combined)

        x1 = self.gat1(x_proj, edge_index, edge_attr)
        x1 = self.bn1(x1)
        x1 = self.elu(x1)
        x1 = self.dropout(x1)
        x1 = x1 + x_proj

        x2 = self.gat2(x1, edge_index, edge_attr)
        x2 = self.bn2(x2)
        x2 = self.elu(x2)
        x2 = self.dropout(x2)
        x2 = x2 + x1

        out = self.fc(x2)
        return out
    
    def get_embedding(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        layer = data.layer

        layer_emb = self.layer_embedding(layer)
        x_combined = torch.cat([x, layer_emb], dim=1)
        x_proj = self.input_proj(x_combined)

        x1 = self.gat1(x_proj, edge_index, edge_attr)
        x1 = self.bn1(x1)
        x1 = self.elu(x1)
        x1 = self.dropout(x1)
        x1 = x1 + x_proj

        x2 = self.gat2(x1, edge_index, edge_attr)
        x2 = self.bn2(x2)
        x2 = self.elu(x2)
        x2 = self.dropout(x2)
        x2 = x2 + x1
        
        return x2.view(-1, self.heads * self.hidden_dim)
    
    def get_layer_embedding(self, layer):
        return self.layer_embedding(layer)

# ------------------------
# 4. Loss function
# ------------------------
class OptimizedFocalLoss(nn.Module):
    def __init__(self, gamma=1.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            n_count = train_counts[0] if 0 in train_counts else 1
            y_count = train_counts[1] if 1 in train_counts else 1
            alpha = torch.tensor([1/n_count, 1/y_count], device=device)
            alpha = alpha / alpha.sum() * 2
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ------------------------
# 5. Training and evaluation
# ------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        loss = loss_fn(out, data.y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * data.num_nodes
        pred = out.argmax(dim=1)
        all_preds.append(pred.cpu())
        all_labels.append(data.y.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    train_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    train_prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    train_recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    avg_loss = total_loss / sum(d.num_nodes for d in loader.dataset)
    return avg_loss, train_f1, train_prec, train_recall

def evaluate_with_threshold(model, loader, loss_fn, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels, all_node_ids = [], [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            node_ids = data.node_id.cpu()
            all_node_ids.append(node_ids)
            all_labels.append(data.y.cpu())
            
            out = model(data)
            loss = loss_fn(out, data.y)
            total_loss += loss.item() * data.num_nodes
            
            prob = F.softmax(out, dim=1)[:, 1]
            all_probs.append(prob.cpu())
    
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    all_node_ids = torch.cat(all_node_ids)
    avg_loss = total_loss / sum(d.num_nodes for d in loader.dataset)
    
    pred = (all_probs >= threshold).float()
    f1 = f1_score(all_labels, pred, average='binary', zero_division=0)
    precision = precision_score(all_labels, pred, average='binary', zero_division=0)
    recall = recall_score(all_labels, pred, average='binary', zero_division=0)
    auc_roc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    auc_pr = average_precision_score(all_labels, all_probs)
    
    return avg_loss, f1, precision, recall, auc_roc, auc_pr, pred, all_labels, all_probs, all_node_ids

def find_optimal_threshold(probs, labels, target_recall=0.8):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    pr_precision = precision[:-1]
    pr_recall = recall[:-1]
    valid_thresholds = thresholds
    
    valid_idx = pr_recall >= target_recall
    if not valid_idx.any():
        valid_idx = pr_recall >= pr_recall.max()
    
    valid_precision = pr_precision[valid_idx]
    valid_recall = pr_recall[valid_idx]
    valid_thresholds = valid_thresholds[valid_idx]
    
    if len(valid_precision) == 0:
        optimal_thresh = 0.5
        best_f1 = 0.0
    else:
        valid_f1 = 2 * (valid_precision * valid_recall) / (valid_precision + valid_recall + 1e-8)
        optimal_idx = np.argmax(valid_f1)
        optimal_thresh = valid_thresholds[optimal_idx]
        best_f1 = valid_f1[optimal_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label='PR Curve (two-stage strategy - Phase 1 + Layer Aware)', color='blue')
    plt.scatter(pr_recall[np.where(valid_thresholds==optimal_thresh)[0][0]], 
                pr_precision[np.where(valid_thresholds==optimal_thresh)[0][0]],
                marker='o', color='red', s=100, label=f'Phase 1 optimal threshold ({optimal_thresh:.2f})')
    plt.xlabel('Recall (target class)')
    plt.ylabel('Precision (target class)')
    plt.title(f'PR Curve (Target Recall >= {target_recall}, Best F1={best_f1:.4f}, two-stage strategy + Layer Aware)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "pr_curve_two_stage_phase1_layer_aware.png"))
    plt.close()
    
    print(f"Phase-1 optimal threshold: {optimal_thresh:.4f}")
    return optimal_thresh

# ------------------------
# 6. Single-model training and prediction
# ------------------------
def train_single_model(input_dim, layer_emb_dim, max_layer, train_loader, val_loader, device):
    print("\n===== Training the Stage 1 model =====")
    model = EdgeTypeAwareGAT(
        input_dim=input_dim,
        layer_emb_dim=layer_emb_dim,
        max_layer=max_layer,
        hidden_dim=hidden_dim,
        num_heads=4,
        dropout=dropout_rate
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    loss_fn = OptimizedFocalLoss(gamma=1.0)
    
    best_val_auc_pr = 0.0
    best_model_path = os.path.join(save_dir, "best_model_layer_aware.pt")
    early_stop_count = 0
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_f1, train_prec, train_recall = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_loss, val_f1, val_prec, val_recall, val_auc_roc, val_auc_pr, _, _, _, _ = evaluate_with_threshold(
            model, val_loader, loss_fn, device
        )
        
        scheduler.step(val_auc_pr)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val AUC-PR: {val_auc_pr:.4f}")
        
        if val_auc_pr > best_val_auc_pr:
            best_val_auc_pr = val_auc_pr
            torch.save(model.state_dict(), best_model_path)
            early_stop_count = 0
        else:
            early_stop_count += 1
        
        if early_stop_count >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(torch.load(best_model_path))
    return model

def single_model_predict(model, loader, device, threshold=0.5):
    all_probs, all_labels, all_node_ids = [], [], []
    
    model.eval()
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            node_ids = data.node_id.cpu()
            all_node_ids.append(node_ids)
            all_labels.append(data.y.cpu())
            
            out = model(data)
            prob = F.softmax(out, dim=1)[:, 1]
            all_probs.append(prob.cpu())
    
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    all_node_ids = torch.cat(all_node_ids)
    
    pred = (all_probs >= threshold).float()
    return pred, all_labels, all_probs, all_node_ids

# ------------------------
# 7. Two-stage feature extraction
# ------------------------
def get_trajectory_stats(data, node_idx):
    return [0.0]*8

def extract_phase2_features(model, loader, device, threshold=0.5):
    all_X = []
    all_y = []
    all_node_ids = []
    
    model.eval()
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            batch_node_ids = data.node_id.cpu().numpy()
            batch_y = data.y.cpu().numpy()
            batch_layer = data.layer.cpu().numpy()
            
            out = model(data)
            phase1_prob_tensor = F.softmax(out, dim=1)[:, 1].cpu()
            phase1_prob_np = phase1_prob_tensor.numpy()
            avg_emb = model.get_embedding(data).cpu().numpy()
            avg_layer_emb = model.get_layer_embedding(data.layer).cpu().numpy()
            
            phase1_y_mask = phase1_prob_np >= threshold
            
            for idx in np.where(phase1_y_mask)[0]:
                node_id = batch_node_ids[idx]
                node_y = batch_y[idx]
                node_layer_val = batch_layer[idx]
                node_layer_emb = avg_layer_emb[idx]
                
                raw_feat = data.x[idx].cpu().numpy()
                gnn_feat = avg_emb[idx]
                traj_feat = get_trajectory_stats(data, idx)
                phase1_feat = [phase1_prob_np[idx]]
                layer_feat = [node_layer_val] + node_layer_emb.tolist()
                veto_feats = compute_veto_features(idx, data.edge_index.cpu(), phase1_prob_tensor)
                
                fused_feat = np.concatenate([
                    raw_feat, gnn_feat, traj_feat, phase1_feat, layer_feat, veto_feats
                ])
                
                all_X.append(fused_feat)
                all_y.append(node_y)
                all_node_ids.append(node_id)
    
    return np.array(all_X), np.array(all_y), np.array(all_node_ids)

# ------------------------
# Main pipeline
# ------------------------
if __name__ == "__main__":
    print("Starting the MSPGL two-stage training pipeline.")
    all_data = load_all_data(data_root)

    if len(all_data) == 0:
        raise ValueError("No PyG graph files were loaded. Please check data_root.")

    print("Splitting the dataset into train/validation/test subsets.")
    train_dataset, val_dataset, test_dataset, train_counts = split_dataset(
        all_data, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    train_dataset = [d for d in train_dataset if d.num_nodes > 0]
    val_dataset = [d for d in val_dataset if d.num_nodes > 0]
    test_dataset = [d for d in test_dataset if d.num_nodes > 0]

    input_dim = train_dataset[0].x.shape[1]
    print(f"Stage-1 input dimension: {input_dim}")

    train_sampler = get_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("\n===== Train Stage 1 model =====")
    model = train_single_model(input_dim, layer_emb_dim, max_layer, train_loader, val_loader, device)

    print("Selecting the Stage-1 threshold on the validation set.")
    val_pred, val_labels, val_probs, _ = single_model_predict(
        model, val_loader, device, 0.5
    )
    optimal_thresh = find_optimal_threshold(val_probs.numpy(), val_labels.numpy(), PHASE1_TARGET_RECALL)

    print("Evaluating Stage-1 predictions on the test set.")
    phase1_test_pred, phase1_test_labels, phase1_test_probs, phase1_test_node_ids = single_model_predict(
        model, test_loader, device, optimal_thresh
    )

    print("Extracting Stage-2 features from the trained Stage-1 model.")
    train_X, train_y, _ = extract_phase2_features(model, train_loader, device, optimal_thresh)
    val_X, val_y, _ = extract_phase2_features(model, val_loader, device, optimal_thresh)
    test_X, test_y, test_node_ids = extract_phase2_features(model, test_loader, device, optimal_thresh)

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    val_X_scaled = scaler.transform(val_X)
    test_X_scaled = scaler.transform(test_X)

    print("Training the single Stage-2 XGBoost model.")
    phase2_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=np.sum(train_y==0)/max(np.sum(train_y==1),1),
        objective='binary:logistic', eval_metric='aucpr', random_state=RANDOM_SEED
    )
    phase2_model.fit(train_X_scaled, train_y, eval_set=[(val_X_scaled, val_y)])

    test_probs = phase2_model.predict_proba(test_X_scaled)[:,1]
    phase2_pred = (test_probs >= PHASE2_THRESHOLD).astype(int)
    np.save(os.path.join(save_dir, "train_X_phase2.npy"), train_X)
    np.save(os.path.join(save_dir, "val_X_phase2.npy"), val_X)
    np.save(os.path.join(save_dir, "test_X_phase2.npy"), test_X)
    np.save(os.path.join(save_dir, "test_y_phase2.npy"), test_y)

    print("Saving Stage-1 and Stage-2 prediction results.")
    phase1_df = pd.DataFrame({
        "node_id": phase1_test_node_ids.numpy(),
        "true_label": phase1_test_labels.numpy(),
        "phase1_pred": phase1_test_pred.numpy(),
        "phase1_prob": phase1_test_probs.numpy()
    })
    phase1_df.to_csv(os.path.join(save_dir, "phase1_result.csv"), index=False)

    phase2_df = pd.DataFrame({
        "node_id": test_node_ids,
        "true_label": test_y,
        "phase2_pred": phase2_pred,
        "phase2_prob": test_probs
    })
    phase2_df.to_csv(os.path.join(save_dir, "phase2_result.csv"), index=False)

    # ==============================================
    # ==============================================
    print("\n===== Save trained models =====")

    torch.save({
        "input_dim": input_dim,
        "max_layer": max_layer,
        "layer_emb_dim": layer_emb_dim,
        "hidden_dim": hidden_dim,
        "optimal_threshold": optimal_thresh,
        "phase2_threshold": PHASE2_THRESHOLD
    }, os.path.join(save_dir, "model_config.pt"))

    joblib.dump(phase2_model, os.path.join(save_dir, "xgb_phase2.model"))

    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

    print("\nAll models have been saved and are ready for inference.")
    print(f"Save directory: {save_dir}")

# %% [markdown] Cell 5
# # Extract target points using the MSPGL model

# %% Cell 6
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax

# ===================== 1. Path Configuration =====================
# Reviewers only need to update these paths when running the inference pipeline.
MODEL_DIR = "./outputs/model"
PT_FOLDER = "./data/pyg_graphs"
OUTPUT_DETAIL = "./outputs/inference/extraction_result.csv"
OUTPUT_SUMMARY = "./outputs/inference/extraction_quality.csv"
OUTPUT_PERFORMANCE = "./outputs/inference/inference_performance_report.csv"
OUTPUT_PERFORMANCE_HTML = "./outputs/inference/inference_performance_report.html"

# The cleaned reviewer version uses one Stage-1 GAT model and removes the legacy leaves feature.
STAGE1_MODEL_FILE = "best_model_layer_aware.pt"


# ===================== 2. Performance Tracking =====================
class PerformanceTracker:
    """Collect device information and per-file inference timing."""

    def __init__(self):
        self.device_info = self._get_device_info()
        self.results = []

    def _get_device_info(self):
        """Return basic information about the current compute device."""
        device_info = {
            "device": "CPU",
            "device_name": "CPU",
            "is_cuda": False,
            "is_mps": False,
            "torch_version": torch.__version__,
        }

        if torch.cuda.is_available():
            device_info["device"] = "CUDA"
            device_info["device_name"] = torch.cuda.get_device_name(0)
            device_info["is_cuda"] = True
            device_info["cuda_version"] = torch.version.cuda
            device_info["num_gpus"] = torch.cuda.device_count()
            device_info["cuda_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_info["device"] = "MPS"
            device_info["device_name"] = "Apple Metal Performance Shaders"
            device_info["is_mps"] = True

        try:
            import platform
            import psutil

            device_info["platform"] = platform.platform()
            device_info["cpu_count"] = psutil.cpu_count()
            device_info["total_memory_gb"] = f"{psutil.virtual_memory().total / 1e9:.2f} GB"
            device_info["available_memory_gb"] = f"{psutil.virtual_memory().available / 1e9:.2f} GB"
        except ImportError:
            pass

        return device_info

    def log_device_info(self):
        """Print device information for reproducibility."""
        print("\n" + "=" * 80)
        print("Compute Device Information")
        print("=" * 80)
        print(f"Primary device: {self.device_info['device']}")
        print(f"Device name: {self.device_info['device_name']}")
        print(f"PyTorch version: {self.device_info['torch_version']}")
        if self.device_info["is_cuda"]:
            print(f"CUDA version: {self.device_info['cuda_version']}")
            print(f"GPU count: {self.device_info['num_gpus']}")
            print(f"GPU memory: {self.device_info['cuda_memory']}")
        if "platform" in self.device_info:
            print(f"Platform: {self.device_info['platform']}")
            print(f"CPU cores: {self.device_info['cpu_count']}")
            print(f"Total memory: {self.device_info['total_memory_gb']}")
            print(f"Available memory: {self.device_info['available_memory_gb']}")
        print("=" * 80 + "\n")

    def add_result(self, file_name, num_nodes, num_edges, y_true_count, y_pred_count,
                   recall, f1, precision, total_time, stages_time):
        """Store the inference summary for a single graph file."""
        self.results.append({
            "file_name": file_name,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "y_true_count": y_true_count,
            "y_pred_count": y_pred_count,
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "total_time_s": round(total_time, 3),
            "time_data_load_ms": round(stages_time["data_load"] * 1000, 2),
            "time_phase1_ms": round(stages_time["phase1_inference"] * 1000, 2),
            "time_feature_extract_ms": round(stages_time["feature_extract"] * 1000, 2),
            "time_phase2_ms": round(stages_time["phase2_inference"] * 1000, 2),
            "time_metrics_ms": round(stages_time["metrics_calc"] * 1000, 2),
            "throughput_nodes_per_sec": round(num_nodes / (total_time + 1e-6), 1),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    def save_results_to_csv(self, output_path):
        """Save the accumulated timing report to CSV."""
        if not self.results:
            print("No performance records were generated.")
            return pd.DataFrame()
        df = pd.DataFrame(self.results)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Performance report saved: {output_path}")
        return df

    def generate_html_report(self, csv_path, html_path):
        """Generate a lightweight HTML summary from the performance CSV."""
        if not os.path.exists(csv_path):
            return
        df = pd.read_csv(csv_path)
        if df.empty:
            return

        total_files = len(df)
        total_time = df["total_time_s"].sum()
        avg_time = df["total_time_s"].mean()
        total_nodes = df["num_nodes"].sum()
        throughput = total_nodes / (total_time + 1e-6)
        slowest = df.nlargest(min(5, len(df)), "total_time_s")

        html_rows = "".join(
            f"<tr><td>{row.file_name}</td><td>{row.num_nodes}</td><td>{row.total_time_s:.3f}</td>"
            f"<td>{row.time_phase1_ms:.2f}</td><td>{row.time_feature_extract_ms:.2f}</td>"
            f"<td>{row.time_phase2_ms:.2f}</td></tr>"
            for row in slowest.itertuples(index=False)
        )
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head><meta charset="UTF-8"><title>MSPGL Inference Performance Report</title></head>
        <body style="font-family: Arial, sans-serif; margin: 24px;">
            <h1>MSPGL Inference Performance Report</h1>
            <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <h2>Summary</h2>
            <ul>
                <li>Total files: {total_files}</li>
                <li>Total time: {total_time:.3f} s</li>
                <li>Average time per file: {avg_time:.3f} s</li>
                <li>Total nodes: {int(total_nodes)}</li>
                <li>Throughput: {throughput:.1f} nodes/s</li>
            </ul>
            <h2>Slowest Files</h2>
            <table border="1" cellpadding="6" cellspacing="0">
                <tr><th>File</th><th>Nodes</th><th>Total Time (s)</th><th>Stage 1 (ms)</th><th>Feature Extraction (ms)</th><th>Stage 2 (ms)</th></tr>
                {html_rows}
            </table>
        </body>
        </html>
        """
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML performance report saved: {html_path}")

    def print_summary(self, df):
        """Print a compact performance summary."""
        if df is None or df.empty:
            return
        print("\n" + "=" * 80)
        print("Inference Performance Summary")
        print("=" * 80)
        print(f"Processed files: {len(df)}")
        print(f"Total time: {df['total_time_s'].sum():.3f} s")
        print(f"Average time: {df['total_time_s'].mean():.3f} s/file")
        print(f"Average Recall: {df['recall'].mean():.4f}")
        print(f"Average F1: {df['f1'].mean():.4f}")
        print(f"Average Precision: {df['precision'].mean():.4f}")
        print("=" * 80 + "\n")


# ===================== 3. Load Model Configuration =====================
config = torch.load(os.path.join(MODEL_DIR, "model_config.pt"), map_location="cpu")
input_dim = config["input_dim"]
max_layer = config["max_layer"]
layer_emb_dim = config["layer_emb_dim"]
hidden_dim = config["hidden_dim"]
optimal_thresh = config["optimal_threshold"]
phase2_thresh = config["phase2_threshold"]


# ===================== 4. Stage-1 GAT Model Definition =====================
class EdgeTypeAwareGATConv(MessagePassing):
    """GAT convolution layer with an edge-type attention bias."""

    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0,
                 edge_type_dim=5, negative_slope=0.2, concat=True, bias=True):
        super().__init__(aggr="add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.edge_type_bias = nn.Parameter(torch.Tensor(edge_type_dim, heads))
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize trainable parameters using the same scheme as training."""
        glorot(self.lin.weight)
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.edge_type_bias)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        h, c = self.heads, self.out_channels
        x = self.lin(x).view(-1, h, c)
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        alpha = (x_i * self.att_dst).sum(-1) + (x_j * self.att_src).sum(-1)
        edge_bias = torch.matmul(edge_attr[:, 2:], self.edge_type_bias)
        alpha = F.leaky_relu(alpha + edge_bias, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(1)
        if self.bias is not None:
            aggr_out += self.bias
        return aggr_out


class EdgeTypeAwareGAT(nn.Module):
    """Single Stage-1 GAT model used to generate candidate target nodes."""

    def __init__(self, input_dim, layer_emb_dim, max_layer, hidden_dim=128,
                 num_heads=4, num_classes=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = num_heads
        self.layer_embedding = nn.Embedding(max_layer + 1, layer_emb_dim)
        self.input_proj = nn.Linear(input_dim + layer_emb_dim, hidden_dim * num_heads)
        self.gat1 = EdgeTypeAwareGATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.bn1 = BatchNorm(hidden_dim * num_heads)
        self.gat2 = EdgeTypeAwareGATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.bn2 = BatchNorm(hidden_dim * num_heads)
        self.fc = nn.Linear(hidden_dim * num_heads, num_classes)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_attr, layer = data.x, data.edge_index, data.edge_attr, data.layer
        layer_emb = self.layer_embedding(layer)
        x = torch.cat([x, layer_emb], dim=1)
        x = self.input_proj(x)
        x1 = self.bn1(self.gat1(x, edge_index, edge_attr))
        x1 = self.dropout(self.elu(x1)) + x
        x2 = self.bn2(self.gat2(x1, edge_index, edge_attr))
        x2 = self.dropout(self.elu(x2)) + x1
        return self.fc(x2)

    def get_embedding(self, data):
        """Return node embeddings used by the Stage-2 XGBoost classifier."""
        x, edge_index, edge_attr, layer = data.x, data.edge_index, data.edge_attr, data.layer
        layer_emb = self.layer_embedding(layer)
        x = torch.cat([x, layer_emb], dim=1)
        x = self.input_proj(x)
        x1 = self.bn1(self.gat1(x, edge_index, edge_attr))
        x1 = self.dropout(self.elu(x1)) + x
        x2 = self.bn2(self.gat2(x1, edge_index, edge_attr))
        x2 = self.dropout(self.elu(x2)) + x1
        return x2.view(-1, self.heads * self.hidden_dim)


# ===================== 5. Load Trained Models =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stage1_model = EdgeTypeAwareGAT(input_dim, layer_emb_dim, max_layer, hidden_dim).to(device)
stage1_path = os.path.join(MODEL_DIR, STAGE1_MODEL_FILE)
stage1_model.load_state_dict(torch.load(stage1_path, map_location=device))
stage1_model.eval()
print(f"Loaded single Stage-1 GAT model: {stage1_path}")

phase2_model = joblib.load(os.path.join(MODEL_DIR, "xgb_phase2.model"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
print("Loaded Stage-2 XGBoost model and feature scaler.")


# ===================== 6. Helper Functions =====================
def remove_legacy_leaves_feature(data):
    """
    Remove the legacy leaves column from old PyG files.
    Current reviewer-facing MSPGL features exclude leaves; old files may still have 12 columns.
    """
    if hasattr(data, "x") and data.x.dim() == 2 and data.x.shape[1] == input_dim + 1:
        data.x = torch.cat([data.x[:, :10], data.x[:, 11:]], dim=1)
    return data


def compute_veto_features(node_id, edge_index, phase1_prob):
    """Compute three neighborhood consistency features for Stage 2."""
    src, dst = edge_index
    neighbor_nodes = torch.cat([dst[src == node_id], src[dst == node_id]]).unique()
    if neighbor_nodes.numel() == 0:
        return [0.0, 0.0, 0.0]

    neighbor_idx = neighbor_nodes.cpu().numpy()
    neighbor_probs = phase1_prob[neighbor_idx]
    current_prob = float(phase1_prob[node_id])
    positive_ratio = float((neighbor_probs > 0.5).mean())
    mean_diff = current_prob - float(neighbor_probs.mean())
    max_diff = current_prob - float(neighbor_probs.max())
    return [positive_ratio, mean_diff, max_diff]


def get_trajectory_stats(data, node_idx):
    """Reserved trajectory statistics. Kept as zeros to match the training feature layout."""
    return [0.0] * 8


def extract_phase2_features(model, data, device, phase1_prob):
    """Build Stage-2 features for nodes retained by the Stage-1 threshold."""
    data = data.to(device)
    with torch.no_grad():
        gat_embedding = model.get_embedding(data).cpu().numpy()
        layer_embedding = model.layer_embedding(data.layer).cpu().numpy()

    feature_rows = []
    valid_idx = []
    for node_idx in range(data.num_nodes):
        if phase1_prob[node_idx] < optimal_thresh:
            continue
        valid_idx.append(node_idx)
        raw_features = data.x[node_idx].cpu().numpy()
        graph_features = gat_embedding[node_idx]
        trajectory_features = get_trajectory_stats(data, node_idx)
        stage1_features = [phase1_prob[node_idx]]
        layer_features = [data.layer[node_idx].item()] + layer_embedding[node_idx].tolist()
        veto_features = compute_veto_features(node_idx, data.edge_index.cpu(), phase1_prob)
        feature_rows.append(np.concatenate([
            raw_features,
            graph_features,
            trajectory_features,
            stage1_features,
            layer_features,
            veto_features,
        ]))

    if not feature_rows:
        return np.empty((0, scaler.n_features_in_)), valid_idx
    return np.array(feature_rows), valid_idx


# ===================== 7. Batch Inference =====================
perf_tracker = PerformanceTracker()
perf_tracker.log_device_info()

all_detail = []
all_summary = []
pt_files = sorted([f for f in os.listdir(PT_FOLDER) if f.endswith(".pt")])
print(f"Found {len(pt_files)} PyG graph files. Starting batch inference...\n")

for file_idx, file_name in enumerate(pt_files, 1):
    graph_path = os.path.join(PT_FOLDER, file_name)
    print(f"===== [{file_idx}/{len(pt_files)}] Processing: {file_name} =====")

    total_start = time.time()
    stages_time = {
        "data_load": 0.0,
        "phase1_inference": 0.0,
        "feature_extract": 0.0,
        "phase2_inference": 0.0,
        "metrics_calc": 0.0,
    }

    try:
        t0 = time.time()
        data = torch.load(graph_path, map_location=device)
        if not isinstance(data, Data):
            print(f"Skip {file_name}: object is not a PyG Data instance.")
            continue
        data = remove_legacy_leaves_feature(data)
        y_true = data.y.cpu().numpy()
        stages_time["data_load"] = time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            data = data.to(device)
            logits = stage1_model(data)
            phase1_probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        stages_time["phase1_inference"] = time.time() - t0

        t0 = time.time()
        phase2_features, valid_idx = extract_phase2_features(stage1_model, data, device, phase1_probs)
        stages_time["feature_extract"] = time.time() - t0

        t0 = time.time()
        y_pred = np.zeros_like(y_true)
        phase2_probs = np.zeros_like(phase1_probs)
        if len(phase2_features) > 0:
            phase2_features = scaler.transform(phase2_features)
            target_probs = phase2_model.predict_proba(phase2_features)[:, 1]
            y_pred[valid_idx] = (target_probs >= phase2_thresh).astype(int)
            phase2_probs[valid_idx] = target_probs
        stages_time["phase2_inference"] = time.time() - t0

        t0 = time.time()
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        stages_time["metrics_calc"] = time.time() - t0

        total_time = time.time() - total_start
        print(f"Recall: {recall:.4f} | F1: {f1:.4f} | Precision: {precision:.4f}")
        print(
            f"Total time: {total_time:.3f} s "
            f"(Stage1: {stages_time['phase1_inference'] * 1000:.1f} ms, "
            f"Features: {stages_time['feature_extract'] * 1000:.1f} ms, "
            f"Stage2: {stages_time['phase2_inference'] * 1000:.1f} ms)"
        )

        perf_tracker.add_result(
            file_name=file_name,
            num_nodes=data.num_nodes,
            num_edges=data.edge_index.size(1) // 2,
            y_true_count=int(y_true.sum()),
            y_pred_count=int(y_pred.sum()),
            recall=recall,
            f1=f1,
            precision=precision,
            total_time=total_time,
            stages_time=stages_time,
        )

        for node_id in range(data.num_nodes):
            all_detail.append({
                "pt_file": file_name,
                "node_id": node_id,
                "y_true": int(y_true[node_id]),
                "y_pred": int(y_pred[node_id]),
                "phase1_prob": float(phase1_probs[node_id]),
                "phase2_prob": float(phase2_probs[node_id]),
                "label": "Y" if y_pred[node_id] == 1 else "N",
            })

        all_summary.append({
            "pt_file": file_name,
            "num_nodes": data.num_nodes,
            "num_positive": int(y_true.sum()),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
        })

    except Exception as exc:
        print(f"Failed to process {file_name}: {exc}")
        continue


# ===================== 8. Save Outputs =====================
os.makedirs(os.path.dirname(OUTPUT_DETAIL), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_SUMMARY), exist_ok=True)

pd.DataFrame(all_detail).to_csv(OUTPUT_DETAIL, index=False)
pd.DataFrame(all_summary).to_csv(OUTPUT_SUMMARY, index=False)

print("\nBatch inference completed.")
print(f"Node-level prediction file: {OUTPUT_DETAIL}")
print(f"Graph-level summary file: {OUTPUT_SUMMARY}")

performance_df = perf_tracker.save_results_to_csv(OUTPUT_PERFORMANCE)
perf_tracker.generate_html_report(OUTPUT_PERFORMANCE, OUTPUT_PERFORMANCE_HTML)
perf_tracker.print_summary(performance_df)

print("\n" + "=" * 80)
print("Output Files")
print("=" * 80)
print(f"Node-level predictions: {OUTPUT_DETAIL}")
print(f"Graph-level metrics: {OUTPUT_SUMMARY}")
print(f"Performance CSV: {OUTPUT_PERFORMANCE}")
print(f"Performance HTML: {OUTPUT_PERFORMANCE_HTML}")
print("=" * 80)

# %% [markdown] Cell 7
# # Robustness analysis

# %% Cell 8
# ==============================================================
# Statistics for Virtual Trajectory Class Imbalance
# Function: Recursively count label distribution (Y/N) of all CSV trajectory files
# Metric: IMB Imbalance Metric
# Formula: IMB = 1 - [ 2 * min(N_yes, N_no) / (N_yes + N_no) ]
# Where:
#    N_yes = Number of target samples (label=Y, browsing target)
#    N_no  = Number of non-target samples (label=N)
# Value range explanation:
#    IMB=0: perfectly balanced (N_yes=N_no)
#    IMB close to 1: extremely imbalanced
# Special Rule: If no target samples (N_yes=0), IMB = 1.0 (full non-target)
# Encoding: UTF-8 
# ==============================================================
import csv
from pathlib import Path

# ===================== Global Configuration =====================
# Root directory of CSV trajectory files (traverse subfolders recursively)
INPUT_FOLDER = "./data/raw_csv"
# Output path of imbalance statistics result
OUTPUT_CSV = "./outputs/imbalance/IMB.csv"
# Column name of label field (Y = browsing target, N = non-target)
LABEL_COLUMN = "MSPGL_label"
LEGACY_LABEL_COLUMN = "label"
# =====================================================================================

def traverse_and_calculate(root_directory: str) -> list:
    """
    Traverse all CSV files recursively, calculate label counts and IMB imbalance metric
    Args:
        root_directory: Root path of CSV files
    Returns:
        list: Statistics of each trajectory file
    """
    stat_records = []
    root = Path(root_directory)
    # Get all CSV files in directory and subdirectories
    csv_file_list = list(root.rglob("*.csv"))
    print(f"Detected total {len(csv_file_list)} CSV files, start statistics...\n")

    for file_path in csv_file_list:
        file_name = file_path.name
        total_points = 0
        target_count = 0    # Count of browsing target (Y, N_yes)
        non_target_count = 0# Count of non-target (N, N_no)

        try:
            # Open CSV file, ignore unreadable characters
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)

                # Check if label column exists
                label_column = LABEL_COLUMN if LABEL_COLUMN in reader.fieldnames else LEGACY_LABEL_COLUMN
                if label_column not in reader.fieldnames:
                    print(f"Warning: {file_name} missing label column, skipped")
                    continue

                # Count labels row by row
                for row in reader:
                    total_points += 1
                    label = row[label_column].strip().upper()
                    if label == "Y":
                        target_count += 1
                    elif label == "N":
                        non_target_count += 1

            # Calculate IMB imbalance metric with given formula
            N_yes = target_count
            N_no = non_target_count
            total = N_yes + N_no
            if total == 0:
                imb_metric = 1.0
            elif N_yes == 0:
                # All samples are non-target, maximum imbalance
                imb_metric = 1.0
            else:
                min_val = min(N_no, N_yes)
                imb_metric = 1 - (2 * min_val) / total
                imb_metric = round(imb_metric, 4)

            # Append record
            stat_records.append([
                file_name,
                total_points,
                target_count,
                non_target_count,
                imb_metric
            ])
            print(f"Success: {file_name} | Total:{total_points} | Target(Y):{target_count} | Non-target(N):{non_target_count} | IMB:{imb_metric}")

        except Exception as err:
            print(f"Error: Failed to process {file_name} | Details: {str(err)}")
            continue

    return stat_records

def export_results(stat_data: list, save_path: str):
    """
    Export statistical results to CSV file
    Args:
        stat_data: List of trajectory statistics
        save_path: Output file path
    """
    # Define CSV header (update imbalance metric column name)
    header = [
        "File Name",
        "Total Trajectory Points",
        "Target Count (Y, N_yes)",
        "Non-target Count (N, N_no)",
        "IMB Imbalance Metric"
    ]
    # Write data to CSV
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(stat_data)

if __name__ == "__main__":
    # Run statistics pipeline
    results = traverse_and_calculate(INPUT_FOLDER)
    # Save final results
    export_results(results, OUTPUT_CSV)

    # Print final summary
    print("\n" + "="*70)
    print("All tasks completed successfully!")
    print(f"Total processed files: {len(results)}")
    print(f"Result file saved at: {OUTPUT_CSV}")
    print("="*70)

# %% [markdown] Cell 9
# # Model interpretability analysis

# %% Cell 10
# ==============================================================
# MSPGL SHAP Analysis Using Saved Trained Models
# Paper: Multi-stage Pyramid Graph Learning for Virtual Trajectory Target Recognition
# Function:
# 1. Load the already trained Stage-2 XGBoost model
# 2. Load saved Phase-2 feature matrices from the training pipeline
# 3. Run SHAP analysis without retraining any model
# ==============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import StandardScaler

# ===================== Path Configuration =====================
# MODEL_INPUT_DIR contains trained artifacts from the training step:
# xgb_phase2.model, scaler.pkl, train_X_phase2.npy, test_X_phase2.npy,
# test_y_phase2.npy, and optional model_config.pt.
MODEL_INPUT_DIR = "./outputs/model"

# SHAP_OUTPUT_DIR stores only SHAP figures, CSV summaries, and the guide file.
SHAP_OUTPUT_DIR = "./outputs/shap"
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

# ===================== Plot Style =====================
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'


# ===================== Load Saved Artifacts =====================
def load_saved_shap_inputs(model_input_dir):
    """
    Load trained Stage-2 model and saved Phase-2 features.
    The SHAP analysis explains the XGBoost classifier, so no GAT/XGBoost retraining is needed here.
    """
    model_path = os.path.join(model_input_dir, "xgb_phase2.model")
    scaler_path = os.path.join(model_input_dir, "scaler.pkl")
    train_x_path = os.path.join(model_input_dir, "train_X_phase2.npy")
    test_x_path = os.path.join(model_input_dir, "test_X_phase2.npy")
    test_y_path = os.path.join(model_input_dir, "test_y_phase2.npy")
    config_path = os.path.join(model_input_dir, "model_config.pt")

    required_files = [model_path, scaler_path, train_x_path, test_x_path, test_y_path]
    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required SHAP input files:\n" + "\n".join(missing))

    phase2_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    train_X = np.load(train_x_path)
    test_X = np.load(test_x_path)
    test_y = np.load(test_y_path)

    config = {}
    if os.path.exists(config_path):
        try:
            import torch
            config = torch.load(config_path, map_location="cpu")
        except Exception as exc:
            print(f"Warning: failed to load model_config.pt: {exc}")

    train_X_scaled = scaler.transform(train_X)
    test_X_scaled = scaler.transform(test_X)
    return phase2_model, train_X_scaled, test_X_scaled, test_y, config


# ===================== Feature Names =====================
def build_phase2_feature_names(n_features, config=None):
    """
    Build readable feature names for the saved Phase-2 matrix.
    Layout follows the training code: raw attribute features + graph embedding
    + Stage-I prediction probability + level embedding + neighborhood consistency features.
    """
    config = config or {}
    raw_dim = int(config.get("input_dim", 11))
    hidden_dim = int(config.get("hidden_dim", 128))
    num_heads = int(config.get("num_heads", 4))
    layer_emb_dim = int(config.get("layer_emb_dim", 32))
    gat_dim = hidden_dim * num_heads

    raw_base_names = [
        "node_index", "normlon", "normlat", "normlevel", "optype",
        "numlevel", "trandistance", "trandirection", "opdistance",
        "standtime", "density"
    ]
    if raw_dim <= len(raw_base_names):
        raw_names = raw_base_names[:raw_dim]
    else:
        raw_names = raw_base_names + [f"raw_extra_{i}" for i in range(raw_dim - len(raw_base_names))]

    feature_names = []
    feature_groups = {}

    start = len(feature_names)
    feature_names.extend(raw_names)
    feature_groups["Raw attribute features"] = list(range(start, len(feature_names)))

    start = len(feature_names)
    feature_names.extend(["GATemb" for _ in range(gat_dim)])
    feature_groups["Graph embedding"] = list(range(start, len(feature_names)))

    feature_names.extend([f"trajectory_stat_{i}" for i in range(8)])

    start = len(feature_names)
    feature_names.append("Stage1Prob")
    feature_groups["Stage-I prediction probability"] = [start]

    start = len(feature_names)
    feature_names.append("raw_level_value")
    feature_names.extend(["levelemb" for _ in range(layer_emb_dim)])
    feature_groups["Level embedding"] = list(range(start, len(feature_names)))

    start = len(feature_names)
    feature_names.extend(["posratio", "meandiff", "maxdiff"])
    feature_groups["Neighborhood consistency features"] = list(range(start, len(feature_names)))

    if len(feature_names) < n_features:
        start = len(feature_names)
        feature_names.extend([f"extra_feature_{i}" for i in range(n_features - len(feature_names))])
        feature_groups["Extra features"] = list(range(start, len(feature_names)))
    elif len(feature_names) > n_features:
        feature_names = feature_names[:n_features]
        feature_groups = {
            name: [idx for idx in idx_list if idx < n_features]
            for name, idx_list in feature_groups.items()
        }
        feature_groups = {name: idx_list for name, idx_list in feature_groups.items() if idx_list}

    return feature_names, feature_groups


def aggregate_shap_features(shap_values, feature_values, feature_names, exclude_prefixes=("trajectory_stat_",), exclude_names=("node_index", "raw_level_value")):
    """
    Aggregate repeated embedding dimensions into reviewer-facing feature names.
    GATemb and levelemb are treated as single features, while trajectory placeholders are excluded.
    """
    shap_values = np.asarray(shap_values)
    feature_values = np.asarray(feature_values)

    aggregate_order = []
    aggregate_indices = {}
    for idx, name in enumerate(feature_names):
        if name in exclude_names or any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        display_name = name
        if name == "GATemb":
            display_name = "GATemb"
        elif name == "levelemb":
            display_name = "levelemb"

        if display_name not in aggregate_indices:
            aggregate_indices[display_name] = []
            aggregate_order.append(display_name)
        aggregate_indices[display_name].append(idx)

    aggregated_shap = []
    aggregated_values = []
    for name in aggregate_order:
        idx_list = aggregate_indices[name]
        aggregated_shap.append(shap_values[:, idx_list].sum(axis=1))
        aggregated_values.append(feature_values[:, idx_list].mean(axis=1))

    return np.column_stack(aggregated_shap), np.column_stack(aggregated_values), aggregate_order


# ===================== Custom SHAP Visualization =====================
def custom_force_plot(base_value, shap_value, feature_values, feature_names, filename, top_k=12):
    """Create a compact force-style plot for one representative sample."""
    shap_value = np.asarray(shap_value)
    feature_values = np.asarray(feature_values)
    top_idx = np.argsort(-np.abs(shap_value))[:top_k]

    ranked = [(feature_names[i], shap_value[i], feature_values[i]) for i in top_idx]
    ranked.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["#d62728" if val > 0 else "#1f77b4" for _, val, _ in ranked]
    labels = [name for name, _, _ in ranked]
    values = [val for _, val, _ in ranked]

    ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value")
    ax.set_title(f"Single-sample SHAP contributions | base={base_value:.4f}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# ===================== SHAP Analysis Pipeline =====================
def run_shap_analysis(phase2_model, train_X_scaled, test_X_scaled, test_y, output_dir, config=None):
    """Run SHAP analysis for the saved Stage-2 XGBoost model."""
    feature_names, feature_groups = build_phase2_feature_names(test_X_scaled.shape[1], config)

    print("\n" + "=" * 70)
    print("          XGBoost SHAP Feature Contribution Analysis")
    print("=" * 70)
    print(f"Samples for SHAP: {test_X_scaled.shape[0]} | Features: {test_X_scaled.shape[1]}")

    explainer = shap.TreeExplainer(phase2_model)
    shap_values = explainer.shap_values(test_X_scaled)
    base_value = explainer.expected_value
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.asarray(base_value).ravel()[-1]

    shap_values_display, test_X_display, feature_names_display = aggregate_shap_features(
        shap_values, test_X_scaled, feature_names
    )
    print(f"SHAP calculation completed | Base value: {float(base_value):.4f}")
    print(f"Displayed features after aggregation: {len(feature_names_display)}")

    # 1. Feature importance bar plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_display, test_X_display, feature_names=feature_names_display, plot_type="bar", show=False, max_display=25)
    plt.title("SHAP Feature Importance Ranking", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_01_importance_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Beeswarm plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_display, test_X_display, feature_names=feature_names_display, plot_type="dot", show=False, max_display=25)
    plt.title("SHAP Feature Contribution Distribution", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_02_beeswarm.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Dependence plots for the top 4 features
    mean_abs_shap = np.abs(shap_values_display).mean(axis=0)
    top4_idx = np.argsort(-mean_abs_shap)[:min(4, len(feature_names_display))]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for ax_idx, feat_idx in enumerate(top4_idx):
        shap.dependence_plot(feat_idx, shap_values_display, test_X_display, feature_names=feature_names_display, ax=axes[ax_idx], show=False)
        axes[ax_idx].set_title(f"Top {ax_idx + 1} Feature: {feature_names_display[feat_idx]}", fontsize=12, fontweight="bold")
    for ax_idx in range(len(top4_idx), len(axes)):
        axes[ax_idx].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_03_dependence_top4.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Single-sample force-style plot
    test_probs = phase2_model.predict_proba(test_X_scaled)[:, 1]
    sample_idx = np.argmax(test_probs) if len(np.where(test_probs >= 0.8)[0]) == 0 else np.where(test_probs >= 0.8)[0][0]
    custom_force_plot(
        float(base_value),
        shap_values_display[sample_idx],
        test_X_display[sample_idx],
        feature_names_display,
        os.path.join(output_dir, "shap_04_force_plot_sample.png"),
    )

    # 5. Misclassification analysis
    test_preds = (test_probs >= 0.5).astype(int)
    fn_idx = np.where((test_preds == 0) & (test_y == 1))[0]
    fp_idx = np.where((test_preds == 1) & (test_y == 0))[0]
    if len(fn_idx) > 0 or len(fp_idx) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        if len(fn_idx) > 0:
            avg_shap_fn = np.abs(shap_values_display[fn_idx]).mean(axis=0)
            top_fn = np.argsort(-avg_shap_fn)[:12]
            axes[0].barh(range(len(top_fn)), avg_shap_fn[top_fn], color="#d62728")
            axes[0].set_yticks(range(len(top_fn)))
            axes[0].set_yticklabels([feature_names_display[i] for i in top_fn])
            axes[0].set_title(f"False Negative (n={len(fn_idx)})", fontweight="bold")
        else:
            axes[0].axis("off")
        if len(fp_idx) > 0:
            avg_shap_fp = np.abs(shap_values_display[fp_idx]).mean(axis=0)
            top_fp = np.argsort(-avg_shap_fp)[:12]
            axes[1].barh(range(len(top_fp)), avg_shap_fp[top_fp], color="#ff7f0e")
            axes[1].set_yticks(range(len(top_fp)))
            axes[1].set_yticklabels([feature_names_display[i] for i in top_fp])
            axes[1].set_title(f"False Positive (n={len(fp_idx)})", fontweight="bold")
        else:
            axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_05_misclassification_analysis.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # 6. Feature group contribution
    group_import = {}
    for name, idx_list in feature_groups.items():
        group_import[name] = np.abs(shap_values[:, idx_list]).mean()
    group_df = pd.DataFrame(list(group_import.items()), columns=["Group", "Mean |SHAP|"]).sort_values("Mean |SHAP|", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(group_df["Group"], group_df["Mean |SHAP|"])
    plt.ylabel("Mean |SHAP value|")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_06_feature_group_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Save statistical results
    feature_import_df = pd.DataFrame({
        "Feature": feature_names_display,
        "Mean |SHAP|": np.abs(shap_values_display).mean(axis=0),
        "Std": np.std(shap_values_display, axis=0),
    }).sort_values("Mean |SHAP|", ascending=False)
    feature_import_df.to_csv(os.path.join(output_dir, "shap_feature_importance.csv"), index=False)
    group_df.to_csv(os.path.join(output_dir, "shap_feature_group_importance.csv"), index=False)

    guide_text = """SHAP Analysis Guide

This SHAP analysis reuses the trained Stage-2 XGBoost model and saved Phase-2 features.
No GAT or XGBoost model is retrained in this section. Feature names are generated from the
saved model configuration so that high-dimensional GAT and layer embeddings are labeled consistently.
"""
    with open(os.path.join(output_dir, "SHAP_Guide.txt"), "w", encoding="utf-8") as f:
        f.write(guide_text)

    print("All SHAP figures and tables were saved successfully.")


# ===================== Main SHAP Execution =====================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STEP 1: Load saved trained model and Phase-2 features")
    phase2_model, train_X_scaled, test_X_scaled, test_y, config = load_saved_shap_inputs(MODEL_INPUT_DIR)

    print("\n" + "=" * 80)
    print("STEP 2: Run SHAP analysis without retraining")
    run_shap_analysis(phase2_model, train_X_scaled, test_X_scaled, test_y, SHAP_OUTPUT_DIR, config)

    print("\n" + "=" * 80)
    print("SHAP analysis completed successfully.")
    print(f"Model inputs loaded from: {MODEL_INPUT_DIR}")
    print(f"SHAP outputs saved to: {SHAP_OUTPUT_DIR}")
    print("=" * 80)

