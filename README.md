# LidarToolkit

## A complete toolkit for LiDAR data processing, visualization, and analysis

![LiDAR Banner](https://github.com/erickfaria/LidarToolkit/raw/main/images/lidarToolkitBanner.jpg)

## Overview

**LidarToolkit** is a Python toolkit for comprehensive processing of LiDAR (Light Detection and Ranging) data. This project offers a complete pipeline for working with 3D point clouds, from initial reading to digital terrain model generation and advanced surface analysis.

Developed for spatial data scientists, geologists, civil engineers, and environmental researchers, LidarToolkit facilitates the extraction of valuable information from LiDAR datasets with a simple and intuitive interface implemented in Jupyter Notebooks.

## Key Features

- **Data Loading and Exploration**: Intuitive interface for selecting and loading LAS/LAZ files
- **Advanced Visualization**:
  - 2D rendering with density control
  - Interactive 3D visualization via Plotly and Open3D
  - Color coding based on attributes (elevation, intensity, classification)

![3D Visualization](https://github.com/erickfaria/LidarToolkit/raw/main/images/3DVisualization.png)

- **DTM (Digital Terrain Model) Generation**:
  - Multiple interpolation methods
  - Customizable resolution
  - Post-processing (gap filling, smoothing)

![DTM Example](https://github.com/erickfaria/LidarToolkit/raw/main/images/mdtInetrpolation.png)

- **Terrain Analysis**:
  - Slope calculation
  - Aspect
  - Hillshade
  - Curvature
  - Contour generation with customizable intervals

![Terrain Analysis](https://github.com/erickfaria/LidarToolkit/raw/main/images/terrain_analysis.png)

- **Data Export**:
  - Support for multiple formats (GeoTIFF, ASC, PNG, XYZ)
  - Preserved metadata and georeferencing

## Project Structure

```
LidarToolkit/
├── lidar/                           # Main package
│   ├── lidarReadAndViewer.py        # Data reading and visualization
│   ├── lidarDTM.py                  # DTM generation and analysis
│   └── ...
├── data/                            # Folder for storing LiDAR files
├── processed_data/                  # Exported results
├── examples/                        # Demonstration Jupyter notebooks
│   │   ├── download_sample.py           # Simple file download example
│   └── download_and_process.py      # Complete workflow example
├── assets/                          # Images and resources
└── README.md                        # This document
```

## Requirements and Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook/Lab

### Installation

```bash
# Clone the repository
git clone https://github.com/erickfaria/LidarToolkit.git
cd LidarToolkit

# Set up virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## How to Use

1. **Data Preparation**
   - Place your LAS/LAZ files in the `data/` folder

2. **Exploration and Visualization**
   - Open the notebook `examples/lidarReadAndViewer.ipynb`
   - Follow the instructions to explore 2D/3D visualizations and attributes

   ![Exploration Example](https://github.com/erickfaria/LidarToolkit/raw/main/images/topViewPointLidar.png)

3. **Digital Terrain Model Generation**
   - Open the notebook `examples/createLidarMDT.ipynb`
   - Adjust parameters to create optimized DTMs
   - Explore derived analyses such as slope, aspect, and hillshade

   ![DTM Example](https://github.com/erickfaria/LidarToolkit/raw/main/images/mdtPostprocessed.png)

## Application Examples

### Topographic Mapping

LidarToolkit enables the creation of detailed topographic maps from LiDAR data. Through DTM generation and contour lines, it's possible to obtain precise terrain representations.

![Topographic Map](https://github.com/erickfaria/LidarToolkit/raw/main/images/Contourmap.png)

### Relief Analysis

Using the terrain analysis features, you can identify important geomorphological characteristics such as steep slopes, plains, and drainage patterns.

![Relief Analysis](https://github.com/erickfaria/LidarToolkit/raw/main/images/curvature.png)

### Advanced Visualization

The 3D visualization tools allow you to interactively explore the point cloud, facilitating the identification of structures and patterns in LiDAR data.

![Advanced Visualization](https://github.com/erickfaria/LidarToolkit/raw/main/images/3DVisualization.png)

## Technologies Used

- **LiDAR Processing**: laspy, pylas
- **Spatial Analysis**: numpy, scipy
- **Visualization**: matplotlib, plotly, Open3D
- **Interactive Interface**: IPython, ipywidgets
- **Geospatial Processing**: rasterio, gdal

## Additional Resources

- [laspy Documentation](https://laspy.readthedocs.io/)
- [Open3D Documentation](http://www.open3d.org/docs/)
- [LAS Standard](https://www.asprs.org/divisions-committees/lidar-division/laser-las-file-format-exchange-activities)

## Roadmap

Features planned for future versions:

- Automatic classification of LiDAR points
- Object detection and extraction (trees, buildings)
- Comparison of multiple LiDAR datasets
- Web interface for remote visualization and processing
- Support for processing very large datasets (> 1 billion points)

## Contributions

Contributions are welcome! If you want to contribute to LidarToolkit:

1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/new-functionality`)
3. Commit your changes (`git commit -m 'Adding new functionality'`)
4. Push to the branch (`git push origin feature/new-functionality`)
5. Open a Pull Request

For issues, suggestions, or questions, please open an issue on GitHub.

## License

This project is licensed under the MIT License.

## Contact

Erick Faria - [GitHub](https://github.com/erickfaria)

Project link: [https://github.com/erickfaria/LidarToolkit](https://github.com/erickfaria/LidarToolkit)

---

<p align="center">
  <i>Developed with ❤️ for the geospatial community</i>
</p>
