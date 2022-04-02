#!/usr/bin/python3
import random
from sys import argv

def main():
    if len(argv) != 3:
        raise Exception("You must provide 'n' and 'm'")
    n = int(argv[1])
    m = int(argv[2])

    # n = 10 # tamanho da primeira sequência
    # m = 20 # tamanho da segunda sequência


    file = 'dna.seq' # nome do arquivo a ser gerado

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

if __name__ == '__main__':
    main()