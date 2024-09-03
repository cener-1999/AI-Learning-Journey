from pathlib import Path
import sys
cur_path = Path.cwd()
root_path = cur_path.parent.parent.parent
sys.path.append(str(root_path))
from settings import set_environment
set_environment()