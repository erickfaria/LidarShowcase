"""
LiDAR Visualization Module.
This module contains utilities for loading, visualizing,
and analyzing LiDAR data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import laspy
import open3d as o3d
import plotly.graph_objects as go
from IPython.display import display
import ipywidgets as widgets


class LidarFileHandler:
    """Class for managing LiDAR files and providing selection widgets."""
    
    @staticmethod
    def list_las_files(data_dir="data"):
        """
        Lists LAS/LAZ files in a directory.
        
        Args:
            data_dir (str): Directory path to search for LiDAR files
            
        Returns:
            tuple: List of LAS/LAZ files and the directory path
        """
        las_files = [f for f in os.listdir(data_dir) if f.endswith(('.las', '.laz'))]
        return las_files, data_dir
    
    @staticmethod
    def create_file_selector(las_files, data_dir):
        """
        Creates a widget for file selection.
        
        Args:
            las_files (list): List of LAS/LAZ files
            data_dir (str): Directory path containing the files
            
        Returns:
            tuple: A dropdown widget and the selected file path
        """
        selected_file_path = os.path.join(data_dir, las_files[0]) if las_files else None
        
        def select_file(change):
            nonlocal selected_file_path
            selected_file_path = os.path.join(data_dir, change['new'])
            print(f"Selected file: {selected_file_path}")
        
        file_dropdown = widgets.Dropdown(
            options=las_files,
            description='Select file:',
            style={'description_width': 'initial'}
        )
        
        file_dropdown.observe(select_file, names='value')
        return file_dropdown, selected_file_path
    
    @staticmethod
    def read_las_file(file_path):
        """
        Reads a LAS/LAZ file.
        
        Args:
            file_path (str): Path to the LAS/LAZ file
            
        Returns:
            laspy.LasData or None: The loaded LiDAR data or None if an error occurs
        """
        try:
            las_data = laspy.read(file_path)
            print("\nFile loaded successfully!")
            return las_data
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    @staticmethod
    def export_subset(las_data, output_path, mask=None, header_changes=None):
        """
        Exports a subset of points to a new LAS file.
        
        Args:
            las_data (laspy.LasData): The LiDAR data
            output_path (str): Path to save the output file
            mask (numpy.ndarray, optional): Boolean mask for point selection
            header_changes (dict, optional): Dictionary of header attribute changes
            
        Returns:
            bool: True if export succeeds, False otherwise
        """
        try:
            # Create a copy of the points
            if mask is None:
                # Export all points
                subset = las_data.copy()
            else:
                # Export only points matching the mask
                subset = las_data.points[mask].copy()
            
            # Apply header changes if any
            if header_changes:
                for key, value in header_changes.items():
                    if hasattr(subset.header, key):
                        setattr(subset.header, key, value)
            
            # Save the file
            subset.write(output_path)
            print(f"File saved successfully at: {output_path}")
            print(f"Total exported points: {subset.header.point_count:,}")
            return True
        except Exception as e:
            print(f"Error exporting file: {e}")
            return False


class LidarInfo:
    """Class for retrieving and displaying information about LiDAR data."""
    
    @staticmethod
    def print_basic_info(las_data):
        """
        Displays basic information about the LAS/LAZ file.
        
        Args:
            las_data (laspy.LasData): The LiDAR data
        """
        if las_data is None:
            return
        
        print(f"LAS format version: {las_data.header.version}")
        print(f"Total point count: {las_data.header.point_count}")
        print(f"Data extent:")
        print(f"  X: {las_data.header.mins[0]:.2f} to {las_data.header.maxs[0]:.2f}")
        print(f"  Y: {las_data.header.mins[1]:.2f} to {las_data.header.maxs[1]:.2f}")
        print(f"  Z: {las_data.header.mins[2]:.2f} to {las_data.header.maxs[2]:.2f}")
        
        # Show available dimensions (attributes) in the data
        print("\nAvailable attributes:")
        for dimension in las_data.point_format.dimensions:
            print(f"  - {dimension.name}")
    
    @staticmethod
    def get_class_names():
        """
        Returns a dictionary with standard LAS class names.
        
        Returns:
            dict: Dictionary mapping class codes to descriptive names
        """
        return {
            0: "Created, never classified",
            1: "Unclassified",
            2: "Ground",
            3: "Low Vegetation",
            4: "Medium Vegetation",
            5: "High Vegetation",
            6: "Building",
            7: "Low Point (noise)",
            8: "Key-point",
            9: "Water",
            10: "Rail",
            11: "Road Surface",
            12: "Overlap Point",
            13: "Wire - Guard",
            14: "Wire - Conductor (phase)",
            15: "Transmission Tower",
            16: "Wire-structure Connector",
            17: "Bridge Deck",
            18: "High Noise"
        }


class LidarVisualization:
    """Class for visualizing LiDAR data."""
    
    @staticmethod
    def get_points(las_data, max_points=None):
        """
        Extracts coordinates and creates a point array.
        
        Args:
            las_data (laspy.LasData): The LiDAR data
            max_points (int, optional): Maximum number of points to extract
            
        Returns:
            numpy.ndarray: Array of points with shape (n, 3)
        """
        # Extract coordinates
        x = las_data.x
        y = las_data.y
        z = las_data.z
        
        # Limit the number of points for faster visualization if necessary
        if max_points and max_points < len(x):
            idx = np.random.choice(len(x), max_points, replace=False)
            x = x[idx]
            y = y[idx]
            z = z[idx]
        
        return np.vstack((x, y, z)).transpose()
    
    @staticmethod
    def plot_2d_view(las_data, max_points=100000, point_size=0.1, point_alpha=0.5, color_map='viridis', use_hexbin=False):
        """
        Creates a 2D (top-down) visualization of the point cloud.
        
        Args:
            las_data (laspy.LasData): The LiDAR data
            max_points (int): Maximum number of points to plot
            point_size (float): Size of points in the scatter plot
            point_alpha (float): Transparency of points (0-1)
            color_map (str): Matplotlib colormap name
            use_hexbin (bool): Whether to use hexbin instead of scatter
        """
        if las_data is None:
            return
        
        plt.figure(figsize=(10, 10))
        # Sample for faster visualization
        sample_idx = np.random.choice(len(las_data.x), min(max_points, len(las_data.x)), replace=False)
        
        if use_hexbin:
            # Visualization using hexbin - creates a continuous representation
            hb = plt.hexbin(las_data.x[sample_idx], las_data.y[sample_idx], 
                           C=las_data.z[sample_idx], 
                           gridsize=100,    # Adjust to control resolution
                           cmap=color_map,
                           mincnt=1,        # Show cells with at least 1 point
                           alpha=point_alpha)
            plt.colorbar(hb, label='Average Elevation (Z)')
        else:
            # Visualization using scatter
            plt.scatter(las_data.x[sample_idx], las_data.y[sample_idx], 
                        c=las_data.z[sample_idx], cmap=color_map, 
                        s=point_size, alpha=point_alpha)
            plt.colorbar(label='Elevation (Z)')
        
        plt.title('Top-Down View of LiDAR Point Cloud')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_3d_plotly(las_data, max_points=200000, point_size=1, opacity=0.8, color_map='Viridis'):
        """
        Creates an interactive 3D visualization using Plotly.
        
        Args:
            las_data (laspy.LasData): The LiDAR data
            max_points (int): Maximum number of points to plot
            point_size (float): Size of points in the 3D plot
            opacity (float): Transparency of points (0-1)
            color_map (str): Plotly colorscale name
            
        Returns:
            plotly.graph_objects.Figure: The interactive 3D figure
        """
        if las_data is None:
            return
        
        # Sample for faster visualization
        sample_size = min(max_points, len(las_data.x))
        sample_idx = np.random.choice(len(las_data.x), sample_size, replace=False)
        
        # Create 3D figure
        fig = go.Figure(data=[go.Scatter3d(
            x=las_data.x[sample_idx],
            y=las_data.y[sample_idx],
            z=las_data.z[sample_idx],
            mode='markers',
            marker=dict(
                size=point_size,
                color=las_data.z[sample_idx],
                colorscale=color_map,
                opacity=opacity,
                colorbar=dict(title="Elevation (Z)")
            )
        )])
        
        fig.update_layout(
            title="3D Visualization of LiDAR Point Cloud",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=900,
            height=700,
        )
        
        return fig
    
    @staticmethod
    def visualize_open3d(las_data, max_points=500000):
        """
        Visualizes the point cloud using Open3D.
        
        Args:
            las_data (laspy.LasData): The LiDAR data
            max_points (int): Maximum number of points to visualize
        """
        if las_data is None:
            return
        
        # Get points (with limit for better performance)
        points = LidarVisualization.get_points(las_data, max_points=max_points)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Normalize heights for coloring
        heights = points[:, 2]  # Z values
        colors = plt.cm.viridis((heights - np.min(heights)) / (np.max(heights) - np.min(heights)))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Instructions
        print("\nOpening Open3D viewer (separate window)...")
        print("Navigation tips:")
        print("- Rotate: left-click and drag")
        print("- Zoom: mouse wheel or right-click + drag")
        print("- Pan: Shift + left-click and drag")
        
        # Visualize
        o3d.visualization.draw_geometries([pcd])
    
    @staticmethod
    def plot_elevation_histogram(las_data):
        """
        Plots a histogram of height (Z) distribution.
        
        Args:
            las_data (laspy.LasData): The LiDAR data
        """
        if las_data is None:
            return
        
        plt.figure(figsize=(12, 6))
        plt.hist(las_data.z, bins=100, alpha=0.7, color='green')
        plt.title('Height (Z) Distribution')
        plt.xlabel('Height (Z)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Basic statistics
        print(f"Height (Z) statistics:")
        print(f"  Minimum: {np.min(las_data.z):.2f}")
        print(f"  Maximum: {np.max(las_data.z):.2f}")
        print(f"  Mean: {np.mean(las_data.z):.2f}")
        print(f"  Median: {np.median(las_data.z):.2f}")
        print(f"  Standard deviation: {np.std(las_data.z):.2f}")
    
    @staticmethod
    def plot_intensity(las_data, max_points=100000):
        """
        Plots intensity distribution and 2D view colored by intensity.
        
        Args:
            las_data (laspy.LasData): The LiDAR data
            max_points (int): Maximum number of points to plot
        """
        if las_data is None or not hasattr(las_data, 'intensity'):
            return
        
        plt.figure(figsize=(12, 10))
        
        # Subplot 1: Intensity histogram
        plt.subplot(2, 1, 1)
        plt.hist(las_data.intensity, bins=100, alpha=0.7, color='purple')
        plt.title('Intensity Value Distribution')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Intensity visualization in point cloud
        plt.subplot(2, 1, 2)
        sample_idx = np.random.choice(len(las_data.x), min(max_points, len(las_data.x)), replace=False)
        plt.scatter(las_data.x[sample_idx], las_data.y[sample_idx], 
                    c=las_data.intensity[sample_idx], cmap='inferno', s=0.1, alpha=0.7)
        plt.colorbar(label='Intensity')
        plt.title('Top-Down View with Intensity Values')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_classification_stats(las_data):
        """
        Plots point classification statistics.
        
        Args:
            las_data (laspy.LasData): The LiDAR data
        """
        if las_data is None or not hasattr(las_data, 'classification'):
            return
        
        class_names = LidarInfo.get_class_names()
        
        # Count points per class
        unique_classes, counts = np.unique(las_data.classification, return_counts=True)
        class_counts = dict(zip(unique_classes, counts))
        
        # Create labels for the chart
        labels = [f"{c} - {class_names.get(c, 'Unknown')} ({class_counts[c]})" for c in unique_classes]
        
        # Plot class distribution
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(unique_classes)), counts, alpha=0.7)
        plt.xticks(range(len(unique_classes)), [str(c) for c in unique_classes], rotation=45)
        plt.title('Point Class Distribution')
        plt.xlabel('Classification Code')
        plt.ylabel('Number of Points')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:,}', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.show()
        
        # Show description of identified classes
        print("\nClasses identified in the dataset:")
        for c in unique_classes:
            percentage = (class_counts[c] / len(las_data.classification)) * 100
            print(f"  Class {c} - {class_names.get(c, 'Unknown')}: {class_counts[c]:,} points ({percentage:.2f}%)")
    
    @staticmethod
    def plot_points_by_class(las_data, class_value, max_points=None, point_size=0.5, title=None):
        """
        Visualizes points with a specific class or list of classes.
        
        Args:
            las_data (laspy.LasData): The LiDAR data
            class_value (int or list): Class value(s) to visualize
            max_points (int, optional): Maximum number of points to plot
            point_size (float): Size of points in the plot
            title (str, optional): Custom title for the plot
            
        Returns:
            tuple: x, y, z coordinates of the filtered points
        """
        # Check if class_value is a list or a single value
        if not isinstance(class_value, list):
            class_value = [class_value]
        
        # Create mask for points of desired classes
        class_mask = np.isin(las_data.classification, class_value)
        
        # Filter points
        x = las_data.x[class_mask]
        y = las_data.y[class_mask]
        z = las_data.z[class_mask]
        
        # Limit number of points if necessary
        if max_points and len(x) > max_points:
            idx = np.random.choice(len(x), max_points, replace=False)
            x = x[idx]
            y = y[idx]
            z = z[idx]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set plot title
        if title:
            plot_title = title
        elif len(class_value) == 1:
            plot_title = f"Points of Class {class_value[0]}"
        else:
            plot_title = f"Points of Classes {', '.join(map(str, class_value))}"
        
        # Create scatter plot with colors based on elevation
        scatter = ax.scatter(x, y, c=z, s=point_size, cmap='viridis', alpha=0.7)
        
        # Add color bar and legends
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Elevation (m)')
        
        ax.set_title(plot_title)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
        # Return filtered points
        return x, y, z