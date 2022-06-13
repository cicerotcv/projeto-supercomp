#include "utils.h"

unsigned int seed = 0;

void show_sequence(char sequence[], int sequence_length) {
  for (int i = 0; i < sequence_length; i++) {
    std::cout << sequence[i];
  }
  std::cout << std::endl;
}

int compare(char a, char b) {
  if (a == b) return WMAT;
  if (a == '-' || b == '-') return WGAP;
  return WMIS;
}

int min(int a, int b) { return a < b ? a : b; }

int max(int a, int b) { return a > b ? a : b; }

int maximo(int a, int b, int c) {
  int max = 0;
  if (a > max) max = a;
  if (b > max) max = b;
  if (c > max) max = c;
  return max;
}

void copy_str(char *dest, char *src, int start, int length) {
  for (int i = 0; i < length; i++) {
    dest[i] = (char)src[start + i];
  }
  dest[length] = '\0';
}

void show_result(std::vector<char> s1, std::vector<char> s2,
                 std::vector<char> resultado) {
  std::vector<char> _s1(s1);
  std::vector<char> _s2(s2);
  std::vector<char> _resultado(resultado);

  std::cout << "Sequência 1: ";
  while (!_s1.empty()) {
    std::cout << _s1.back();
    _s1.pop_back();
  }
  std::cout << std::endl;

  std::cout << "Matches    : ";
  while (!_resultado.empty()) {
    std::cout << _resultado.back();
    _resultado.pop_back();
  }
  std::cout << std::endl;

  std::cout << "Sequência 2: ";
  while (!_s2.empty()) {
    std::cout << _s2.back();
    _s2.pop_back();
  }
  std::cout << std::endl;
}

int simple_score(std::string s1, std::string s2) {
  int score = 0;
  int limit = min(s1.size(), s2.size());
  for (int pos = 0; pos < limit; pos++) {
    score += compare(s1.at(pos), s2.at(pos));
  }
  return score;
}