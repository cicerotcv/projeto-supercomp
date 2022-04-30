# -*- encoding :: UTF-8 -*-

import os
from tools import Result, generate_input, get_executables, get_input_files, get_input_size, run_executable
import json


def generate_input_files(limit):
    """Gera arquivos de entrada em lote"""
    for i in range(1, limit):
        generate_input(i, i)


def main():

    # gera arquivos de entrada de tamanho 60
    generate_input_files(61)

    input_files = get_input_files('entradas')
    project_folders = ['01-heuristica', '02-busca-local', '03-busca-exaustiva']
    
    # lista de arquivos executáveis
    executables = get_executables(project_folders, 'script')

    results = []

    # executa todos os arquivos de entrada em todos os algoritmos
    for exe in executables:
        for input_file in input_files:
            input_size = get_input_size(input_file)
            proc, execution_time = run_executable(exe, input_file)
            result = Result(os.path.dirname(exe), input_size, execution_time)
            results.append(result)
            print(result)

    # formata e salva os um JSON com resultados
    groups = {}
    for result in results:
        group_name = result.executable
        if group_name not in groups:
            groups[group_name] = []
        item = { "input_size": result.input_size, "exec_time": result.exec_time }
        groups[group_name].append(item)

    with open("results.json", 'w+', encoding='utf-8') as output_file:
        output_file.write(json.dumps(groups))

if __name__ == '__main__':
    main()