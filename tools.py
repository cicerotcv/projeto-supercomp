import os
import subprocess
import time
import random

def generate_input(n, m):
    file = f"entradas/{str(n).zfill(4)}x{str(m).zfill(4)}.seq"

    f = open(file, 'w')

    seq = [
        str(n)+'\n',
        str(m)+'\n',
        ''.join(random.choices(['A','T','C','G','-'],k=n)) + '\n',
        ''.join(random.choices(['A','T','C','G','-'],k=m)) + '\n'
    ]

    f.writelines(seq)

    f.close()

    print(''.join(seq))

def get_input_files(input_dir='entradas'):
    input_paths = [ os.path.join(input_dir, input_filename) for input_filename in os.listdir(input_dir) ]
    input_paths.sort()
    return input_paths

def get_executables(folders=[], executable_name='script'):
    executables = [os.path.join(folder, executable_name) for folder in folders]
    executables.sort()
    return executables

def get_input_size(input_file):
    with open(input_file, 'r') as file:
        return int(file.readlines()[0])


class Result:
    def __init__(self, executable, input_size, exec_time):
        self.executable = executable
        self.input_size = input_size
        self.exec_time = exec_time
        self.__str__ = self.__repr__

    def __repr__(self):
        return f"[{self.executable}][size:{self.input_size}] Tempo total(s): {self.exec_time} s"

def run_executable(executable_path:str, input_path:str):
    with open(input_path) as f:
        start = time.perf_counter()
        input = f.read()
        proc = subprocess.run([executable_path], input=input, text=True, capture_output=subprocess.DEVNULL)
        end = time.perf_counter()

        return proc, end - start
