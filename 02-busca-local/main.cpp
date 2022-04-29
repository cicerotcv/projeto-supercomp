#include "utils.h"

int main() {
  int N, M;

  std::cin >> N;
  std::cin >> M;
  std::cout << "N: " << N << ", M: " << M << std::endl;
  std::cout << "WMAT: " << WMAT << std::endl;
  std::cout << "WMIS: " << WMIS << std::endl;
  std::cout << "WGAP: " << WGAP << std::endl;
  std::cout << std::endl;

  char a[N + 1], b[M + 1];

  std::cin >> a;
  std::cin >> b;

  if (max(M, N) <= 100) {
    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
  }

  int k = random_integer(1, min(M, N));
  std::cout << "k: " << k << std::endl;

  int j = random_integer(0, M - k);
  std::cout << "j: " << j << std::endl;

  int start = j;
  int size = k;
  char sb[size + 1];

  copy_str(sb, b, start, size);
  std::cout << std::endl;

  int p = random_integer(1, N - k + 1);
  std::cout << "p: " << p << std::endl << std::endl;

  char s1[k + 1];
  char s2[k + 1];
  int score = 0;

  for (int round = 0; round < p; round++) {
    int i = random_integer(0, N - k);

    int start = i;
    int size = k;
    char sa[size + 1];

    copy_str(sa, a, start, size);

    int current_score = calcula_busca_local(sa, size, sb, size);

    if (current_score >= score) {
      score = current_score;
      copy_str(s1, sa, 0, k);
      copy_str(s2, sb, 0, k);
    }

    // std::cout << "Score: " << current_score << std::endl;
    // std::cout << std::endl;
  }

  std::cout << "Melhor Score: " << score << std::endl;
  std::cout << "Sequência 1 : " << s1 << std::endl;
  std::cout << "Sequência 2 : " << s2 << std::endl;
  return 0;
}