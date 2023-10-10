
import os
import sys
import subprocess
import importlib

def p_import(module_name, package_name=None):
  if package_name is None:
    package_name = module_name

  env_target = os.path.join(os.path.dirname(__file__), '.py-env')
  os.makedirs(env_target, mode=0o777, exist_ok=True)
  if not env_target in sys.path:
    sys.path.append(env_target)

  try:
    return importlib.import_module(module_name)
  except:
    subprocess.run([
      sys.executable, '-m', 'pip', 'install', f'--target={env_target}', *(package_name.split())
    ])

  return importlib.import_module(module_name)


