#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

typedef struct Node Node;

struct Node {
  int value;
  int i;
  int j;
  Node *previous;
};

int compare(char a, char b) {
  // std::cout << a << " " << b << std::endl;

  if (a == b) {
    return 2;
  }

  if (a == '-' || b == '-') {
    return -1;
  }

  return -1;
}

void show_sequence(char sequence[], int sequence_length) {
  for (int i = 0; i < sequence_length; i++) {
    std::cout << sequence[i];
  }
  std::cout << std::endl;
}

int maximo(int a, int b, int c) {
  int max = 0;

  if (a > max) {
    max = a;
  }

  if (b > max) {
    max = b;
  }

  if (c > max) {
    max = c;
  }

  return max;
}

int main() {
  int n, m;

  // n é o tamanho da primeira sequência
  // m é o tamanho da segunda

  std::cin >> n;
  std::cin >> m;
  std::cout << "n: " << n << ", m: " << m << std::endl;

  char a[n], b[m];

  for (int i = 0; i < n; i++) {
    std::cin >> a[i];
  }

  if (n <= 100) {
    std::cout << "Sequência 1: ";
    show_sequence(a, n);
  }

  for (int i = 0; i < m; i++) {
    std::cin >> b[i];
  }

  if (m <= 100) {
    std::cout << "Sequência 2: ";
    show_sequence(b, m);
  }

  Node H[n + 1][m + 1];

  for (int i = 0; i <= n; i++) {
    H[i][0] = {0, i, 0};
  }

  for (int j = 0; j <= m; j++) {
    H[0][j] = {0, 0, j};
  }

  std::cout << std::endl;

  Node max = {0, 0, 0};
  Node current, *upper, *left, *upper_left;

  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= m; j++) {
      upper = &(H[i - 1][j]);
      left = &(H[i][j - 1]);
      upper_left = &(H[i - 1][j - 1]);

      int diagonal = upper_left->value + compare(a[i - 1], b[j - 1]);
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

  // std::cout << std::endl;

  // for (int row = 0; row <= n; row++) {
  //   // std::cout << "[" << row << "]\t";

  //   for (int col = 0; col <= m; col++) {
  //     Node current = H[row][col];
  //     std::cout << current.value << ' ';
  //   }

  //   std::cout << std::endl;
  // }

  // std::cout << std::endl;
  std::cout << std::endl
            << "Máximo: " << max.value << std::endl
            << "i: " << max.i << " j: " << max.j << std::endl;

  current = max;

  std::vector<char> s1;
  std::vector<char> resultado;
  std::vector<char> s2;

  while (current.value != 0) {
    char c1 = a[current.i - 1];
    char c2 = b[current.j - 1];

    char current_char = c1 == '-' || c2 == '-' ? ' ' : c1 == c2 ? '*' : '-';

    c1 = current.previous->i == current.i ? '-' : c1;
    c2 = current.previous->j == current.j ? '-' : c2;

    s1.push_back(c1);
    resultado.push_back(current_char);
    s2.push_back(c2);

    current = *current.previous;
  }

  std::cout << std::endl;

  std::cout << "Sequência 1: ";
  while (!s1.empty()) {
    std::cout << s1.back();
    s1.pop_back();
  }
  std::cout << std::endl;

  std::cout << "Matches    : ";
  while (!resultado.empty()) {
    std::cout << resultado.back();
    resultado.pop_back();
  }
  std::cout << std::endl;

  std::cout << "Sequência 2: ";
  while (!s2.empty()) {
    std::cout << s2.back();
    s2.pop_back();
  }
  std::cout << std::endl;

  return 0;
}