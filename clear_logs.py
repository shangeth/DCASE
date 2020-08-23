import shutil
import os

if __name__ == "__main__":
    path = 'logging'
    shutil.rmtree(path)
    os.mkdir(path)