import os
import argparse
import sys
import subprocess

# Importar do novo pacote lidar
from lidar.downloader import download_files_parallel
from lidar.processor import LazProcessor, ensure_lazrs_installed

def check_dependencies():
    """Verificar e instalar dependências necessárias"""
    required_packages = ["laspy", "tqdm", "requests"]
    
    print("Verificando dependências...")
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Verificar backend LAZ
    ensure_lazrs_installed()

def main():
    # Verificar dependências primeiro
    check_dependencies()
    
    # Configurar argumentos
    parser = argparse.ArgumentParser(description="Baixar e converter arquivos LAZ para LAS")
    parser.add_argument("--input", "-i", help="Arquivo de texto contendo URLs, uma por linha")
    parser.add_argument("--output", "-o", default="./data", help="Pasta de saída para arquivos baixados e convertidos")
    parser.add_argument("--download-workers", "-dw", type=int, default=5, help="Número de downloads paralelos")
    parser.add_argument("--process-workers", "-pw", type=int, default=1, help="Número de processos para conversão")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescrever arquivos existentes")
    args = parser.parse_args()
    
    data_dir = args.output
    os.makedirs(data_dir, exist_ok=True)
    
    # Identificar URLs para download
    if args.input:
        with open(args.input, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        # URLs padrão
        urls = [
            # "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/NE_Statewide_D23/NE_Statewide_3_D23/LAZ/USGS_LPC_NE_Statewide_D23_13T_GH_2362.laz",
            # "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/NE_Statewide_D23/NE_Statewide_3_D23/LAZ/USGS_LPC_NE_Statewide_D23_13T_GH_3843.laz",
            # "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/NE_Statewide_D23/NE_Statewide_3_D23/LAZ/USGS_LPC_NE_Statewide_D23_13T_FG_7085.laz",
        ]
    
    # Passo 1: Baixar arquivos LAZ
    print("=== Etapa 1: Baixando arquivos LAZ ===")
    downloaded = download_files_parallel(urls, data_dir, args.download_workers, args.overwrite)
    print(f"Baixados {len(downloaded)} arquivos para {data_dir}")
    
    # Passo 2: Converter arquivos LAZ para LAS
    print("\n=== Etapa 2: Convertendo arquivos LAZ para LAS ===")
    processor = LazProcessor(data_dir)
    processor.process_all_files(output_dir=data_dir, workers=args.process_workers)
    
    print("\nProcessamento completo! Os arquivos LAS estão disponíveis em:", data_dir)

if __name__ == "__main__":
    main()
