#include "math.h"
#include "utils.h"

typedef struct {
  int length;
  std::string value;
} Sequence;

int get_length(char *string) {
  int length = 0;
  while (string[length] != '\0' && length < 10000) {
    length++;
  }
  return length;
}

void generate_subsequences(std::vector<Sequence> *destination,
                           std::string sequence) {
  const int N = sequence.length();
  for (int length = 1; length <= N; length++) {
    for (int pos = 0; pos <= N - length; pos++) {
      std::string value(sequence.substr(pos, length));
      Sequence sequence = {length, value};
      destination->push_back(sequence);
    }
  }
}

void show_all_subsequences(const std::vector<Sequence> vector) {
  for (Sequence s : vector) {
    std::cout << "[" << s.value << "] ";
  }
  std::cout << std::endl;
}

int main() {
  int N, M;

  std::cin >> N;
  std::cin >> M;
  std::cout << "N: " << N << ", M: " << M << std::endl;
  std::cout << "WMAT: " << WMAT << std::endl;
  std::cout << "WMIS: " << WMIS << std::endl;
  std::cout << "WGAP: " << WGAP << std::endl;
  std::cout << std::endl;

  std::string a;
  std::string b;

  std::cin >> a;
  std::cin >> b;

  std::cout << "a: " << a << std::endl;
  std::cout << "b: " << b << std::endl;

  std::vector<Sequence> subsequences_n;
  std::vector<Sequence> subsequences_m;

  // gera todas as subsequências de tamanho 1 até N
  generate_subsequences(&subsequences_n, a);
  // gera todas as subsequências de tamanho 1 até M
  generate_subsequences(&subsequences_m, b);

  // exibe as subsequências geradas
  std::cout << "Sequência:[N " << subsequences_n.size() << "] ";
  show_all_subsequences(subsequences_n);

  std::cout << std::endl;

  std::cout << "Sequência:[M " << subsequences_m.size() << "] ";
  show_all_subsequences(subsequences_m);

  return 0;
}