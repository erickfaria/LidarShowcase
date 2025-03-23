"""
LiDAR Digital Terrain Model (DTM) Module.
This module contains utilities for creating Digital Terrain Models from LiDAR data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, Rbf
import rasterio
from rasterio.transform import from_origin
import laspy
import geopandas as gpd
import json
from datetime import datetime

# Importações condicionais
try:
    import pdal
    PDAL_AVAILABLE = True
except ImportError:
    PDAL_AVAILABLE = False

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

from skimage import filters, morphology
from rasterio.plot import show


class DTMGenerator:
    """Class for creating Digital Terrain Models from LiDAR data."""
    
    def __init__(self, las_data=None, resolution=1.0):
        """
        Initialize the DTM Generator.
        
        Args:
            las_data (laspy.LasData, optional): LiDAR data to use
            resolution (float): Grid resolution in same units as LiDAR data
        """
        self.las_data = las_data
        self.resolution = resolution
        self.ground_mask = None
        self.dtm = None
        self.dtm_metadata = {}
        self.output_dir = "output"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def set_data(self, las_data):
        """
        Set the LiDAR data to use.
        
        Args:
            las_data (laspy.LasData): LiDAR data to use
        """
        self.las_data = las_data
    
    def set_resolution(self, resolution):
        """
        Set the grid resolution.
        
        Args:
            resolution (float): Grid resolution in same units as LiDAR data
        """
        self.resolution = resolution
    
    def get_ground_points(self, class_value=2):
        """
        Extract ground points from the LAS data.
        
        Args:
            class_value (int): Classification code for ground points (default: 2)
            
        Returns:
            tuple: x, y, z arrays of ground points
        """
        if self.las_data is None:
            raise ValueError("No LiDAR data loaded")
        
        self.ground_mask = self.las_data.classification == class_value
        ground_x = self.las_data.x[self.ground_mask]
        ground_y = self.las_data.y[self.ground_mask]
        ground_z = self.las_data.z[self.ground_mask]
        
        print(f"Extracted {len(ground_x):,} ground points out of {len(self.las_data.x):,} total points")
        
        return ground_x, ground_y, ground_z
    
    def generate_simple_dtm(self, method='linear'):
        """
        Generate a simple DTM using basic interpolation.
        
        Args:
            method (str): Interpolation method: 'linear', 'nearest', 'cubic'
            
        Returns:
            dict: Dictionary containing grid_x, grid_y, and grid_z arrays
        """
        # Extract ground points
        ground_x, ground_y, ground_z = self.get_ground_points()
        
        if len(ground_x) == 0:
            raise ValueError("No ground points found in the data")
        
        # Create a grid for interpolation
        x_min, x_max = np.min(ground_x), np.max(ground_x)
        y_min, y_max = np.min(ground_y), np.max(ground_y)
        
        # Calculate grid dimensions
        x_range = int((x_max - x_min) / self.resolution) + 1
        y_range = int((y_max - y_min) / self.resolution) + 1
        
        print(f"Creating grid with dimensions: {x_range} x {y_range}")
        
        # Create grid coordinates
        grid_x, grid_y = np.mgrid[x_min:x_max:complex(0, x_range), 
                                  y_min:y_max:complex(0, y_range)]
        
        # Reshape points for griddata
        points = np.column_stack((ground_x, ground_y))
        
        # Interpolate heights on the grid
        print(f"Interpolating with method: {method}")
        grid_z = griddata(points, ground_z, (grid_x, grid_y), method=method)
        
        # Store the result
        self.dtm = {
            'grid_x': grid_x,
            'grid_y': grid_y,
            'grid_z': grid_z,
            'resolution': self.resolution,
            'method': f'simple_{method}'
        }
        
        self.dtm_metadata = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'resolution': self.resolution,
            'x_size': x_range,
            'y_size': y_range
        }
        
        print(f"DTM created with grid size: {x_range} x {y_range}")
        return self.dtm
    
    def generate_rbf_dtm(self, function='thin_plate', smooth=0.1):
        """
        Generate a DTM using Radial Basis Function interpolation.
        
        Args:
            function (str): RBF function: 'thin_plate', 'multiquadric', 'gaussian'
            smooth (float): Smoothing parameter
            
        Returns:
            dict: Dictionary containing grid_x, grid_y, and grid_z arrays
        """
        # Extract ground points
        ground_x, ground_y, ground_z = self.get_ground_points()
        
        if len(ground_x) == 0:
            raise ValueError("No ground points found in the data")
        
        # Create a grid for interpolation
        x_min, x_max = np.min(ground_x), np.max(ground_x)
        y_min, y_max = np.min(ground_y), np.max(ground_y)
        
        # Use a smaller subset for RBF training to improve performance
        if len(ground_x) > 20000:
            print(f"Sampling {20000:,} points for RBF fitting...")
            idx = np.random.choice(len(ground_x), 20000, replace=False)
            train_x = ground_x[idx]
            train_y = ground_y[idx]
            train_z = ground_z[idx]
        else:
            train_x = ground_x
            train_y = ground_y
            train_z = ground_z
            
        # Calculate grid dimensions
        x_range = int((x_max - x_min) / self.resolution) + 1
        y_range = int((y_max - y_min) / self.resolution) + 1
        
        print(f"Creating grid with dimensions: {x_range} x {y_range}")
        
        # Create grid coordinates
        grid_x, grid_y = np.mgrid[x_min:x_max:complex(0, x_range), 
                                  y_min:y_max:complex(0, y_range)]
        
        # Create and train RBF model
        print(f"Training RBF with function: {function}, smoothing: {smooth}")
        rbf = Rbf(train_x, train_y, train_z, function=function, smooth=smooth)
        
        # Predict heights on the grid
        print("Predicting heights for the entire grid...")
        grid_z = rbf(grid_x, grid_y)
        
        # Store the result
        self.dtm = {
            'grid_x': grid_x,
            'grid_y': grid_y,
            'grid_z': grid_z,
            'resolution': self.resolution,
            'method': f'rbf_{function}'
        }
        
        self.dtm_metadata = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'resolution': self.resolution,
            'x_size': x_range,
            'y_size': y_range
        }
        
        print(f"RBF DTM created with grid size: {x_range} x {y_range}")
        return self.dtm
    
    def generate_pdal_dtm(self):
        """
        Generate a DTM using PDAL pipeline with more advanced filtering.
        
        Returns:
            numpy.ndarray: The DTM raster data
        """
        if not PDAL_AVAILABLE:
            print("PDAL is not available. Please install it with:")
            print("pip install python-pdal")
            print("\nUsing alternative method (RBF interpolation) instead...")
            return self.generate_rbf_dtm(function='thin_plate', smooth=0.1)
        
        # Save ground points to temporary LAS file
        temp_dir = os.path.join(self.output_dir, "temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        input_file = self.las_data.filename
        if input_file is None:
            # If filename is not available, save a temporary file
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            temp_file = os.path.join(temp_dir, f"temp_input_{timestamp}.las")
            self.las_data.write(temp_file)
            input_file = temp_file
        
        output_file = os.path.join(temp_dir, "ground_filtered.tif")
        
        # Define PDAL pipeline for DTM creation
        pipeline = [
            {
                "type": "readers.las",
                "filename": input_file
            },
            {
                "type": "filters.range",
                "limits": "Classification[2:2]"  # Select ground points only
            },
            {
                "type": "filters.assign",
                "assignment": "Classification[:]=2"  # Ensure all selected points are ground
            },
            {
                "type": "writers.gdal",
                "filename": output_file,
                "output_type": "idw",  # Use inverse distance weighting
                "resolution": str(self.resolution),
                "window_size": 6  # Size of IDW search window
            }
        ]
        
        # Execute pipeline
        print("Running PDAL pipeline for DTM generation...")
        pipeline_json = json.dumps(pipeline)
        pdal_pipeline = pdal.Pipeline(pipeline_json)
        pdal_pipeline.execute()
        
        # Read the output raster
        with rasterio.open(output_file) as src:
            grid_z = src.read(1)
            transform = src.transform
            
            # Get grid coordinates from the transform
            x_size, y_size = src.width, src.height
            x_min = transform[2]
            y_max = transform[5]
            x_max = x_min + x_size * transform[0]
            y_min = y_max + y_size * transform[4]
            
            grid_x, grid_y = np.mgrid[x_min:x_max:complex(0, x_size), 
                                      y_min:y_max:complex(0, y_size)]
        
        # Store the result
        self.dtm = {
            'grid_x': grid_x,
            'grid_y': grid_y,
            'grid_z': grid_z,
            'resolution': self.resolution,
            'method': 'pdal_idw'
        }
        
        self.dtm_metadata = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'resolution': self.resolution,
            'x_size': x_size,
            'y_size': y_size,
            'transform': transform
        }
        
        print(f"PDAL DTM created with grid size: {x_size} x {y_size}")
        return grid_z
    
    def apply_post_processing(self, fill_nodata=True, smoothing=None):
        """
        Apply post-processing to the DTM.
        
        Args:
            fill_nodata (bool): Fill NoData (NaN) values
            smoothing (str, optional): Smoothing method ('gaussian', 'median')
            
        Returns:
            numpy.ndarray: Processed grid_z array
        """
        if self.dtm is None:
            raise ValueError("No DTM available for post-processing")
            
        grid_z = self.dtm['grid_z'].copy()
        
        # Fill NoData values
        if fill_nodata:
            if np.any(np.isnan(grid_z)):
                print("Filling NoData values...")
                # Use nearest neighbor to fill holes
                mask = np.isnan(grid_z)
                grid_x, grid_y = self.dtm['grid_x'], self.dtm['grid_y']
                points = np.column_stack((grid_x[~mask].ravel(), grid_y[~mask].ravel()))
                values = grid_z[~mask].ravel()
                
                xi = np.column_stack((grid_x[mask].ravel(), grid_y[mask].ravel()))
                if len(xi) > 0:
                    filled_values = griddata(points, values, xi, method='nearest')
                    grid_z[mask] = filled_values
        
        # Apply smoothing
        if smoothing:
            print(f"Applying {smoothing} smoothing...")
            if smoothing == 'gaussian':
                grid_z = filters.gaussian(grid_z, sigma=1)
            elif smoothing == 'median':
                grid_z = filters.median(grid_z, morphology.disk(1))
        
        # Update the DTM
        self.dtm['grid_z'] = grid_z
        self.dtm['method'] += f"_processed"
        
        return grid_z
    
    def export_dtm(self, format_list=None, base_filename=None):
        """
        Export the DTM to various formats.
        
        Args:
            format_list (list): List of formats: 'tif', 'asc', 'png', 'xyz'
            base_filename (str, optional): Base name for output files
            
        Returns:
            list: List of output filenames
        """
        if self.dtm is None:
            raise ValueError("No DTM available for export")
            
        if format_list is None:
            format_list = ['tif', 'png']
            
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            base_filename = f"dtm_{self.dtm['method']}_{timestamp}"
        
        output_files = []
        
        for fmt in format_list:
            if fmt == 'tif':
                output_file = self._export_geotiff(base_filename)
                output_files.append(output_file)
            elif fmt == 'asc':
                output_file = self._export_asc(base_filename)
                output_files.append(output_file)
            elif fmt == 'png':
                output_file = self._export_png(base_filename)
                output_files.append(output_file)
            elif fmt == 'xyz':
                output_file = self._export_xyz(base_filename)
                output_files.append(output_file)
        
        return output_files
    
    def _export_geotiff(self, base_filename):
        """Export DTM as GeoTIFF."""
        output_file = os.path.join(self.output_dir, f"{base_filename}.tif")
        
        grid_z = self.dtm['grid_z']
        
        # Convert to float32 for better compatibility
        grid_z_float = grid_z.astype('float32')
        
        # Replace NaN with NoData value
        grid_z_float = np.where(np.isnan(grid_z_float), -9999, grid_z_float)
        
        x_min = self.dtm_metadata.get('x_min')
        y_max = self.dtm_metadata.get('y_max')
        if x_min is None or y_max is None:
            x_min = self.dtm['grid_x'].min()
            y_max = self.dtm['grid_y'].max()
        
        transform = from_origin(x_min, y_max, self.resolution, self.resolution)
        
        # Export to GeoTIFF
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=grid_z.shape[0],
            width=grid_z.shape[1],
            count=1,
            dtype=grid_z_float.dtype,
            crs='+proj=utm +zone=23 +south +datum=WGS84 +units=m +no_defs',  # Example CRS
            transform=transform,
            nodata=-9999
        ) as dst:
            dst.write(grid_z_float, 1)
        
        print(f"Exported DTM to GeoTIFF: {output_file}")
        return output_file
    
    def _export_asc(self, base_filename):
        """Export DTM as ASCII grid."""
        output_file = os.path.join(self.output_dir, f"{base_filename}.asc")
        
        grid_z = self.dtm['grid_z']
        x_min = self.dtm_metadata.get('x_min')
        y_min = self.dtm_metadata.get('y_min')
        
        if x_min is None or y_min is None:
            x_min = self.dtm['grid_x'].min()
            y_min = self.dtm['grid_y'].min()
        
        # Create ASCII grid header
        header = (
            f"ncols {grid_z.shape[1]}\n"
            f"nrows {grid_z.shape[0]}\n"
            f"xllcorner {x_min}\n"
            f"yllcorner {y_min}\n"
            f"cellsize {self.resolution}\n"
            f"NODATA_value -9999"
        )
        
        # Replace NaN with NoData value
        grid_z_export = np.where(np.isnan(grid_z), -9999, grid_z)
        
        # Save to ASC file
        np.savetxt(output_file, grid_z_export, header=header, comments='', fmt='%.3f')
        
        print(f"Exported DTM to ASCII grid: {output_file}")
        return output_file
    
    def _export_png(self, base_filename):
        """Export DTM as PNG image."""
        output_file = os.path.join(self.output_dir, f"{base_filename}.png")
        
        grid_z = self.dtm['grid_z']
        
        # Create a nice visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create shaded relief
        grid_z_filled = np.where(np.isnan(grid_z), np.nanmedian(grid_z), grid_z)
        im = ax.imshow(grid_z_filled, cmap='terrain', vmin=np.nanpercentile(grid_z, 1), 
                       vmax=np.nanpercentile(grid_z, 99))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Elevation (m)')
        
        ax.set_title(f'Digital Terrain Model ({self.dtm["method"]})')
        ax.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Exported DTM visualization to PNG: {output_file}")
        return output_file
    
    def _export_xyz(self, base_filename):
        """Export DTM as XYZ point cloud."""
        output_file = os.path.join(self.output_dir, f"{base_filename}.xyz")
        
        grid_x = self.dtm['grid_x']
        grid_y = self.dtm['grid_y']
        grid_z = self.dtm['grid_z']
        
        # Flatten arrays
        x = grid_x.flatten()
        y = grid_y.flatten()
        z = grid_z.flatten()
        
        # Filter out NaN values
        valid = ~np.isnan(z)
        x = x[valid]
        y = y[valid]
        z = z[valid]
        
        # Combine into XYZ array and export
        xyz = np.column_stack((x, y, z))
        np.savetxt(output_file, xyz, fmt='%.3f', delimiter=' ', header='X Y Z', comments='')
        
        print(f"Exported DTM as XYZ point cloud: {output_file}")
        return output_file
    
    def visualize_dtm(self, add_hillshade=True, colormap='terrain'):
        """
        Visualize the DTM with optional hillshade.
        
        Args:
            add_hillshade (bool): Whether to add hillshade effect
            colormap (str): Matplotlib colormap name
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        if self.dtm is None:
            raise ValueError("No DTM available for visualization")
        
        grid_z = self.dtm['grid_z']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Fill NaN values for visualization
        grid_z_vis = np.copy(grid_z)
        if np.any(np.isnan(grid_z_vis)):
            median_value = np.nanmedian(grid_z_vis)
            grid_z_vis = np.where(np.isnan(grid_z_vis), median_value, grid_z_vis)
        
        # Create hillshade if requested
        if add_hillshade:
            # Calculate hillshade
            dx, dy = np.gradient(grid_z_vis)
            slope = np.pi/2. - np.arctan(np.sqrt(dx*dx + dy*dy))
            aspect = np.arctan2(-dx, dy)
            
            # Azimuth and altitude of light source
            azimuth = np.pi/4.
            altitude = np.pi/4.
            
            # Calculate hillshade
            shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
            
            # Blend hillshade with DTM
            rgb = plt.cm.get_cmap(colormap)((grid_z_vis - np.nanmin(grid_z_vis)) / 
                                          (np.nanmax(grid_z_vis) - np.nanmin(grid_z_vis)))
            
            # Make a shaded version by reducing RGB values according to hillshade
            rgb[..., :3] = rgb[..., :3] * shaded[..., np.newaxis]
            
            # Display the result
            ax.imshow(rgb)
        else:
            # Display DTM without hillshade
            im = ax.imshow(grid_z_vis, cmap=colormap, vmin=np.nanpercentile(grid_z, 2), 
                          vmax=np.nanpercentile(grid_z, 98))
            plt.colorbar(im, ax=ax, label='Elevation (m)')
        
        ax.set_title(f'Digital Terrain Model ({self.dtm["method"]})\nResolution: {self.resolution}m')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def compare_dtms(self, other_dtm, title='DTM Comparison'):
        """
        Compare this DTM with another one visually.
        
        Args:
            other_dtm (dict): Another DTM from this class
            title (str): Title for the comparison plot
            
        Returns:
            matplotlib.figure.Figure: The comparison figure
        """
        if self.dtm is None:
            raise ValueError("No primary DTM available for comparison")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # First DTM
        grid_z1 = self.dtm['grid_z']
        im1 = ax1.imshow(grid_z1, cmap='terrain', vmin=np.nanpercentile(grid_z1, 2), 
                        vmax=np.nanpercentile(grid_z1, 98))
        plt.colorbar(im1, ax=ax1, label='Elevation (m)')
        ax1.set_title(f'DTM 1: {self.dtm["method"]}')
        ax1.axis('off')
        
        # Second DTM
        grid_z2 = other_dtm['grid_z']
        im2 = ax2.imshow(grid_z2, cmap='terrain', vmin=np.nanpercentile(grid_z2, 2), 
                        vmax=np.nanpercentile(grid_z2, 98))
        plt.colorbar(im2, ax=ax2, label='Elevation (m)')
        ax2.set_title(f'DTM 2: {other_dtm["method"]}')
        ax2.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def compute_difference(self, other_dtm):
        """
        Compute the difference between two DTMs.
        
        Args:
            other_dtm (dict): Another DTM from this class
            
        Returns:
            numpy.ndarray: Difference grid
        """
        if self.dtm is None:
            raise ValueError("No primary DTM available for comparison")
            
        grid_z1 = self.dtm['grid_z']
        grid_z2 = other_dtm['grid_z']
        
        # Check if grids have the same shape
        if grid_z1.shape != grid_z2.shape:
            raise ValueError("DTMs have different grid shapes and cannot be compared directly")
            
        # Compute difference
        diff = grid_z1 - grid_z2
        
        # Visualize difference
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Use diverging colormap for difference
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label='Elevation Difference (m)')
        
        ax.set_title(f'DTM Difference: {self.dtm["method"]} - {other_dtm["method"]}')
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        mean_diff = np.nanmean(diff)
        std_diff = np.nanstd(diff)
        min_diff = np.nanmin(diff)
        max_diff = np.nanmax(diff)
        
        print(f"Difference statistics:")
        print(f"Mean difference: {mean_diff:.3f} m")
        print(f"Standard deviation: {std_diff:.3f} m")
        print(f"Min difference: {min_diff:.3f} m")
        print(f"Max difference: {max_diff:.3f} m")
        
        return diff


class TerrainAnalysis:
    """Class for terrain analysis on DTMs."""
    
    @staticmethod
    def calculate_slope(dtm, resolution=None):
        """
        Calculate slope from a DTM.
        
        Args:
            dtm (dict): DTM dictionary from DTMGenerator
            resolution (float, optional): Grid resolution if not in DTM
            
        Returns:
            numpy.ndarray: Slope in degrees
        """
        grid_z = dtm['grid_z']
        
        if resolution is None:
            resolution = dtm.get('resolution', 1.0)
        
        # Calculate gradients
        dy, dx = np.gradient(grid_z, resolution)
        
        # Convert to slope in degrees
        slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))
        
        return slope
    
    @staticmethod
    def calculate_aspect(dtm):
        """
        Calculate aspect from a DTM.
        
        Args:
            dtm (dict): DTM dictionary from DTMGenerator
            
        Returns:
            numpy.ndarray: Aspect in degrees (0=North, 90=East, 180=South, 270=West)
        """
        grid_z = dtm['grid_z']
        
        # Calculate gradients
        dy, dx = np.gradient(grid_z)
        
        # Calculate aspect in radians
        aspect_rad = np.arctan2(dy, dx)
        
        # Convert to compass degrees
        aspect_deg = np.degrees(aspect_rad)
        aspect = 90.0 - aspect_deg
        aspect = np.where(aspect < 0, aspect + 360.0, aspect)
        
        return aspect
    
    @staticmethod
    def create_hillshade(dtm, azimuth=315, altitude=45):
        """
        Create hillshade from a DTM.
        
        Args:
            dtm (dict): DTM dictionary from DTMGenerator
            azimuth (float): Azimuth angle of light in degrees
            altitude (float): Altitude angle of light in degrees
            
        Returns:
            numpy.ndarray: Hillshade values
        """
        grid_z = dtm['grid_z']
        
        # Calculate gradients
        dx, dy = np.gradient(grid_z)
        
        # Calculate slope and aspect in radians
        slope_rad = np.arctan(np.sqrt(dx*dx + dy*dy))
        aspect_rad = np.arctan2(-dx, dy)
        
        # Convert light directions to radians
        azimuth_rad = np.radians(azimuth)
        altitude_rad = np.radians(altitude)
        
        # Calculate hillshade
        hillshade = np.sin(altitude_rad) * np.sin(slope_rad) + \
                   np.cos(altitude_rad) * np.cos(slope_rad) * \
                   np.cos(azimuth_rad - aspect_rad)
        
        # Scale to 0-255 for visualization
        hillshade = 255 * (hillshade + 1) / 2
        
        return hillshade
    
    @staticmethod
    def calculate_curvature(dtm):
        """
        Calculate curvature from a DTM.
        
        Args:
            dtm (dict): DTM dictionary from DTMGenerator
            
        Returns:
            tuple: Profile curvature and plan curvature
        """
        grid_z = dtm['grid_z']
        
        # Calculate first and second derivatives
        dx, dy = np.gradient(grid_z)
        dxx, dxy = np.gradient(dx)
        dyx, dyy = np.gradient(dy)
        
        # Calculate curvature components
        p = dx**2 + dy**2
        q = p + 1
        
        # Profile curvature (curvature in the direction of slope)
        profile_curv = ((dx**2 * dxx + 2 * dx * dy * dxy + dy**2 * dyy) / 
                         (p * np.sqrt(q)))
        
        # Plan curvature (curvature perpendicular to the direction of slope)
        plan_curv = ((dx**2 * dyy - 2 * dx * dy * dxy + dy**2 * dxx) / 
                      (p**1.5))
        
        return profile_curv, plan_curv
    
    @staticmethod
    def visualize_terrain_attributes(dtm, attribute='slope', title=None, colormap=None):
        """
        Visualize various terrain attributes derived from a DTM.
        
        Args:
            dtm (dict): DTM dictionary from DTMGenerator
            attribute (str): Attribute to visualize ('slope', 'aspect', 'hillshade', 'curvature')
            title (str, optional): Custom title for the plot
            colormap (str, optional): Matplotlib colormap name
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        if attribute == 'slope':
            data = TerrainAnalysis.calculate_slope(dtm)
            if colormap is None:
                colormap = 'YlOrRd'
            if title is None:
                title = 'Slope (degrees)'
                
        elif attribute == 'aspect':
            data = TerrainAnalysis.calculate_aspect(dtm)
            if colormap is None:
                colormap = 'twilight'
            if title is None:
                title = 'Aspect (degrees)'
                
        elif attribute == 'hillshade':
            data = TerrainAnalysis.create_hillshade(dtm)
            if colormap is None:
                colormap = 'gray'
            if title is None:
                title = 'Hillshade'
                
        elif attribute == 'curvature':
            profile_curv, plan_curv = TerrainAnalysis.calculate_curvature(dtm)
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # Visualize profile curvature
            vmin = np.nanpercentile(profile_curv, 2)
            vmax = np.nanpercentile(profile_curv, 98)
            im1 = ax1.imshow(profile_curv, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            plt.colorbar(im1, ax=ax1, label='Profile Curvature')
            ax1.set_title('Profile Curvature')
            ax1.axis('off')
            
            # Visualize plan curvature
            vmin = np.nanpercentile(plan_curv, 2)
            vmax = np.nanpercentile(plan_curv, 98)
            im2 = ax2.imshow(plan_curv, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            plt.colorbar(im2, ax=ax2, label='Plan Curvature')
            ax2.set_title('Plan Curvature')
            ax2.axis('off')
            
            plt.suptitle('Terrain Curvature Analysis', fontsize=16)
            plt.tight_layout()
            return fig
            
        else:
            raise ValueError(f"Unknown attribute: {attribute}")
        
        # Visualize the selected attribute
        fig, ax = plt.subplots(figsize=(12, 10))
        
        vmin = np.nanpercentile(data, 2)
        vmax = np.nanpercentile(data, 98)
        
        im = ax.imshow(data, cmap=colormap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=title)
        
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        return fig


class ContourGenerator:
    """Class for generating contour lines from DTMs."""
    
    @staticmethod
    def generate_contours(dtm, interval=5, base=0, smoothing=None):
        """
        Generate contour lines from a DTM.
        
        Args:
            dtm (dict): DTM dictionary from DTMGenerator
            interval (float): Contour interval
            base (float): Base contour value
            smoothing (str, optional): Apply smoothing before contouring ('gaussian', 'median')
            
        Returns:
            matplotlib.contour.QuadContourSet: Contour set
        """
        grid_x = dtm['grid_x']
        grid_y = dtm['grid_y']
        grid_z = dtm['grid_z']
        
        # Apply smoothing if requested
        if smoothing:
            if smoothing == 'gaussian':
                grid_z = filters.gaussian(grid_z, sigma=1)
            elif smoothing == 'median':
                grid_z = filters.median(grid_z, morphology.disk(1))
        
        # Determine contour levels
        z_min = np.floor(np.nanmin(grid_z) / interval) * interval
        z_max = np.ceil(np.nanmax(grid_z) / interval) * interval
        levels = np.arange(z_min, z_max + interval, interval)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Generate contours
        contour_set = ax.contour(grid_x, grid_y, grid_z, levels=levels, colors='black', 
                                 linewidths=0.5)
        
        # Add labels to contours
        ax.clabel(contour_set, inline=True, fontsize=8, fmt='%1.0f')
        
        # Add filled contours for better visualization
        contourf = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap='terrain', alpha=0.7)
        plt.colorbar(contourf, ax=ax, label='Elevation (m)')
        
        ax.set_title(f'Contour Map (Interval: {interval}m)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return contour_set
    
    @staticmethod
    def export_contours_shapefile(dtm, output_file, interval=5, base=0):
        """
        Export contour lines to a shapefile.
        
        Args:
            dtm (dict): DTM dictionary from DTMGenerator
            output_file (str): Output shapefile path
            interval (float): Contour interval
            base (float): Base contour value
            
        Returns:
            str: Output shapefile path
        """
        if not GDAL_AVAILABLE:
            print("GDAL is not available for contour generation to shapefile.")
            print("Please install GDAL with: pip install GDAL")
            return None
            
        # Extract data
        grid_x = dtm['grid_x']
        grid_y = dtm['grid_y']
        grid_z = dtm['grid_z']
        
        # Create temporary GeoTIFF file for contour generation
        temp_dir = os.path.dirname(output_file)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        temp_raster = os.path.join(temp_dir, "temp_raster_for_contours.tif")
        
        x_min = grid_x.min()
        y_max = grid_y.max()
        resolution = dtm.get('resolution', 1.0)
        
        # Create GeoTIFF
        with rasterio.open(
            temp_raster,
            'w',
            driver='GTiff',
            height=grid_z.shape[0],
            width=grid_z.shape[1],
            count=1,
            dtype=grid_z.dtype,
            crs='+proj=utm +zone=23 +south +datum=WGS84 +units=m +no_defs',  # Example CRS
            transform=from_origin(x_min, y_max, resolution, resolution),
            nodata=-9999
        ) as dst:
            dst.write(np.where(np.isnan(grid_z), -9999, grid_z), 1)
        
        # Generate contours using GDAL
        print(f"Generating contours with interval: {interval}m")
        output_shapefile = output_file
        
        # Determine contour levels
        z_min = np.floor(np.nanmin(grid_z) / interval) * interval
        fixed_levels = list(np.arange(z_min, np.nanmax(grid_z) + interval, interval))
        
        # Generate contours using GDAL
        options = gdal.ContourGenerateOptions(
            contourInterval=interval,
            fixedLevels=fixed_levels,
            idField='ID',
            elevField='Elevation',
            noDataValue=-9999
        )
        
        # Open raster dataset
        ds = gdal.Open(temp_raster)
        if ds:
            band = ds.GetRasterBand(1)
            
            # Create output vector dataset
            driver = gdal.GetDriverByName('ESRI Shapefile')
            dst_ds = driver.Create(output_shapefile, 0, 0, 0, gdal.GDT_Unknown)
            
            # Generate contours
            gdal.ContourGenerate(band, interval, base, fixed_levels, 0, -9999, dst_ds, 
                                0, 1, 'Elevation', 0)
            
            # Clean up
            ds = None
            dst_ds = None
            
            print(f"Contours exported to: {output_shapefile}")
            return output_shapefile
        else:
            print("Failed to open raster dataset")
            return None
``` 
