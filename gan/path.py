import os

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.abspath(__file__),
    os.pardir,
    os.pardir
))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
EXP_PATH = os.path.join(DATA_PATH, 'exp')
LOG_PATH = os.path.join(DATA_PATH, 'log')
