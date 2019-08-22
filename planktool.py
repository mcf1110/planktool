import sys
import os

sys.path.append('./src')

import build_dataset
import build_models
import subprocess

if len(sys.argv) <= 1:
    print('No command passed!')
    exit
command = sys.argv[1]

def get_path(p):
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), p))

if command == 'build-dataset':
    build_dataset.build_dataset()
elif command == 'build-models':
    build_models.build_models()
elif command == 'build':
    build_dataset.build_dataset()
    build_models.build_models()
elif command == 'web':
    p = get_path('./src/ui/web')
    subprocess.call("cd %s & flask run" % p, shell=True)
elif command == 'gui':
    p = get_path('./src/ui/gui/main.py')
    subprocess.call("python " + p, shell=True)
else:
    print('Unrecognized command.')