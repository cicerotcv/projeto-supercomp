#include <omp.h>

#include "utils.h"

typedef struct {
  int score;
  std::string seq1;
  std::string seq2;
} Result;

int calcula_busca_local(const std::string sa, const std::string sb) {
  const int length_sa = sa.length();
  const int length_sb = sb.length();

  Node H[length_sa + 1][length_sb + 1];

  for (int i = 0; i <= length_sa; i++) {
    H[i][0] = {0, i, 0};
  }

  for (int j = 0; j <= length_sb; j++) {
    H[0][j] = {0, 0, j};
  }

  Node max = {0, 0, 0};
  Node current, *upper, *left, *upper_left;

  for (int i = 1; i <= length_sa; i++) {
    for (int j = 1; j <= length_sb; j++) {
      upper = &(H[i - 1][j]);
      left = &(H[i][j - 1]);
      upper_left = &(H[i - 1][j - 1]);

      int diagonal = upper_left->value + compare(sa.at(i - 1), sb.at(j - 1));
      int delecao = upper->value - 1;
      int insercao = left->value - 1;

      int current_value = maximo(diagonal, delecao, insercao);

      current = {current_value, i, j};

      if (current_value == diagonal) {
        current.previous = upper_left;
      } else if (current.value == delecao) {
        current.previous = upper;
      } else if (current.value == insercao) {
        current.previous = left;
      }

      H[i][j] = current;

      if (current.value >= max.value) {
        max = current;
      }
    }
  }

  current = max;

  std::vector<char> s1;
  std::vector<char> resultado;
  std::vector<char> s2;

  while (current.value != 0) {
    char c1 = sa.at(current.i - 1);
    char c2 = sb.at(current.j - 1);

    char current_char = c1 == '-' || c2 == '-' ? ' ' : c1 == c2 ? '*' : '-';

    c1 = current.previous->i == current.i ? '-' : c1;
    c2 = current.previous->j == current.j ? '-' : c2;

    s1.push_back(c1);
    resultado.push_back(current_char);
    s2.push_back(c2);

    current = *current.previous;
  }

  return max.value;
}

int same_size(std::string sa, std::string sb) {
  if (sa.size() == sb.size()) {
    return simple_score(sa, sb);
  }
  return 0;
}

void calculate_score(std::vector<Sequence> set_a, std::vector<Sequence> set_b,
                     int (*method)(std::string sa, std::string sb),
                     Result* result) {
  for (Sequence sa : set_a) {
    for (Sequence sb : set_b) {
      int score = method(sa.value, sb.value);
      if (score > result->score) {
        result->score = score;
        result->seq1 = sa.value;
        result->seq2 = sb.value;
      }
    }
  }
}

void generate_subsequences(std::vector<Sequence>* destination,
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

void run() {
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

  if (max(M, N) <= 100) {
    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
  }

  std::vector<Sequence> sn;
  std::vector<Sequence> sm;

  // gera todas as subsequências de tamanho 1 até N
  generate_subsequences(&sn, a);

  // gera todas as subsequências de tamanho 1 até M
  generate_subsequences(&sm, b);

  Result first_method = {0};
  Result second_method = {0};

  // // Heurística de Alinhamento Local
  calculate_score(sm, sn, calcula_busca_local, &first_method);

  // Comparação Simples
  calculate_score(sm, sn, same_size, &second_method);

  Result* result;
  if (first_method.score > second_method.score) {
    result = &first_method;
  } else {
    result = &second_method;
  }

  std::cout << "Max score: " << result->score << std::endl;
  std::cout << "Sequences: " << std::endl
            << "\t" << result->seq1 << std::endl
            << "\t" << result->seq2 << std::endl;
}

int main() {
  clock_t before = std::clock();
  run();
  clock_t after = std::clock();

  double delta_time = (double)(after - before) / CLOCKS_PER_SEC;

  std::cout << "Elapsed time: " << delta_time << std::endl;
  return 0;
}