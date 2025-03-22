"""
LidarShowcase - Ferramentas para download e processamento de dados LiDAR
"""

__version__ = '0.1.0'

from .downloader import download_files_parallel, download_file
from .processor import LazProcessor, ensure_lazrs_installed
