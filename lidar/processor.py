import os
import laspy
import concurrent.futures
from tqdm import tqdm
import subprocess
import sys

def ensure_lazrs_installed():
    """
    Verifica e instala a biblioteca lazrs necessária para descompressão LAZ
    
    Returns:
        bool: True se a instalação foi bem-sucedida, False caso contrário
    """
    try:
        import lazrs
        return True
    except ImportError:
        print("Instalando bibliotecas necessárias para descompressão LAZ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lazrs"])
        try:
            import lazrs
            return True
        except ImportError:
            print("Falha ao instalar lazrs. Tentando pylas...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pylas"])
                return True
            except:
                return False

class LazProcessor:
    """
    Classe para processamento de arquivos LAZ (compactados) para LAS
    
    Args:
        data_dir (str): Diretório contendo arquivos LAZ
    """
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def get_laz_files(self):
        """
        Return all LAZ files in the data directory
        
        Returns:
            list: Lista com caminhos completos para arquivos LAZ
        """
        return [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                if f.lower().endswith('.laz')]
    
    def process_file(self, file_path, output_dir=None):
        """
        Convert a single LAZ file to LAS format
        
        Args:
            file_path (str): Caminho para o arquivo LAZ
            output_dir (str, optional): Diretório de saída para o arquivo LAS
            
        Returns:
            dict: Dicionário com informações sobre o processamento
        """
        try:
            # Set output directory to same as input if not specified
            if output_dir is None:
                output_dir = os.path.dirname(file_path)
            
            # Make sure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Get filename without extension
            base_name = os.path.basename(file_path)
            name_no_ext = os.path.splitext(base_name)[0]
            
            # Configurar backend explicitamente (após a instalação de lazrs)
            try:
                import lazrs
                laspy.LasReader.default_backend = 'lazrs'
            except ImportError:
                # Tentar com o modo alternativo
                pass
            
            # Read the LAZ file and convert to LAS
            with laspy.open(file_path) as f:
                las = f.read()
                
                # Save as LAS
                las_path = os.path.join(output_dir, f"{name_no_ext}.las")
                las.write(las_path)
            
            return {
                "file": base_name,
                "status": "success",
                "output": las_path
            }
        
        except Exception as e:
            return {
                "file": os.path.basename(file_path),
                "status": "error",
                "error": str(e)
            }
    
    def process_all_files(self, output_dir=None, workers=4):
        """
        Process all LAZ files using multiple workers
        
        Args:
            output_dir (str, optional): Diretório de saída para os arquivos LAS
            workers (int): Número de processos paralelos
            
        Returns:
            list: Lista de resultados do processamento
        """
        # Verificar dependências primeiro
        if not ensure_lazrs_installed():
            print("ERRO: Não foi possível instalar as dependências necessárias para descompressão LAZ.")
            print("Tente instalar manualmente com: pip install lazrs pylas")
            return []
            
        files = self.get_laz_files()
        if not files:
            print("Nenhum arquivo LAZ encontrado no diretório:", self.data_dir)
            return []
            
        results = []
        
        # Usar 1 worker para evitar problemas de concorrência com o backend LAZ
        effective_workers = 1 if workers > 1 else workers
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = [executor.submit(self.process_file, f, output_dir) for f in files]
            
            # Process with progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc="Convertendo LAZ para LAS"):
                results.append(future.result())
        
        # Summarize results
        successful = sum(1 for r in results if r["status"] == "success")
        print(f"Convertidos com sucesso {successful} de {len(files)} arquivos")
        
        # Log errors
        errors = [r for r in results if r["status"] == "error"]
        if errors:
            print(f"Erros ocorreram em {len(errors)} arquivos:")
            for e in errors:
                print(f"  {e['file']}: {e['error']}")
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converter arquivos LAZ para LAS")
    parser.add_argument("--input", "-i", default="./data", help="Diretório contendo arquivos LAZ")
    parser.add_argument("--output", "-o", default="./data", help="Diretório de saída para arquivos LAS")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Número de processos trabalhadores")
    args = parser.parse_args()
    
    processor = LazProcessor(args.input)
    processor.process_all_files(output_dir=args.output, workers=args.workers)