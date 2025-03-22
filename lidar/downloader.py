import os
import requests
import concurrent.futures
from tqdm import tqdm
import argparse

def download_file(url, output_folder, overwrite=False):
    """
    Download a file from a URL to a specified folder
    
    Args:
        url (str): URL do arquivo para download
        output_folder (str): Pasta onde o arquivo será salvo
        overwrite (bool): Se True, sobrescreve arquivos existentes
        
    Returns:
        str or None: Caminho para o arquivo baixado ou None se ocorrer erro
    """
    # Extract filename from URL
    filename = os.path.basename(url)
    output_path = os.path.join(output_folder, filename)
    
    # Skip if file exists and we're not overwriting
    if os.path.exists(output_path) and not overwrite:
        print(f"Skipping {filename} (already exists)")
        return None
    
    # Make request and download file
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        file_size = int(response.headers.get('content-length', 0))
        
        # Download with progress
        with open(output_path, 'wb') as f, tqdm(
                desc=filename,
                total=file_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                progress_bar.update(len(chunk))
                
        return output_path
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)  # Remove partial download
        return None

def download_files_parallel(url_list, output_folder, max_workers=5, overwrite=False):
    """
    Download multiple files in parallel
    
    Args:
        url_list (list): Lista de URLs para download
        output_folder (str): Pasta onde os arquivos serão salvos
        max_workers (int): Número máximo de downloads simultâneos
        overwrite (bool): Se True, sobrescreve arquivos existentes
        
    Returns:
        list: Lista de caminhos para os arquivos baixados com sucesso
    """
    os.makedirs(output_folder, exist_ok=True)
    
    downloaded_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(download_file, url, output_folder, overwrite): url 
            for url in url_list
        }
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                file_path = future.result()
                if file_path:
                    downloaded_files.append(file_path)
            except Exception as e:
                print(f"Error processing {url}: {e}")
    
    return downloaded_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LAZ files from URLs")
    parser.add_argument("--input", "-i", help="Text file containing URLs, one per line")
    parser.add_argument("--output", "-o", default="./data", help="Output folder for downloaded files")
    parser.add_argument("--workers", "-w", type=int, default=5, help="Number of parallel downloads")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()
    
    # Read URLs from file
    if args.input:
        with open(args.input, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        # Hardcoded URLs
        urls = [
            "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/NE_Statewide_D23/NE_Statewide_3_D23/LAZ/USGS_LPC_NE_Statewide_D23_13T_GH_2362.laz",
            "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/NE_Statewide_D23/NE_Statewide_3_D23/LAZ/USGS_LPC_NE_Statewide_D23_13T_GH_3843.laz",
            "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/NE_Statewide_D23/NE_Statewide_3_D23/LAZ/USGS_LPC_NE_Statewide_D23_13T_FG_7085.laz",
        ]
    
    # Ensure data directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Download files
    downloaded = download_files_parallel(urls, args.output, args.workers, args.overwrite)
    print(f"Downloaded {len(downloaded)} files to {args.output}")