from pathlib import Path

def find_project_root_dir(folder_name: str = "dady") -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / folder_name).is_dir():
            return parent
    raise 
# CORRECTION : Utiliser uniquement pathlib
ROOT_DIR = find_project_root_dir("dady") / "dady"

SQUARE_PATCHS_DIR = ROOT_DIR / "testdata" / "224x224_patchs"