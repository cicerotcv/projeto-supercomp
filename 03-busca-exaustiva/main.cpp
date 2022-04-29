#include "utils.h"

typedef struct {
  int score;
  std::string seq1;
  std::string seq2;
} Result;

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

  // Heurística de Alinhamento Local
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

  return 0;
}