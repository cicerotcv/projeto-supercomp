#include "utils.h"

unsigned int seed = 0;

void show_sequence(char sequence[], int sequence_length) {
  for (int i = 0; i < sequence_length; i++) {
    std::cout << sequence[i];
  }
  std::cout << std::endl;
}

void get_random_number() {
  std::srand(seed != 0 ? seed : (unsigned int)time(NULL));
  seed = std::rand();
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  double random_number = distribution(generator);
  std::cout << "Random number: " << random_number << std::endl;
}

int random_integer(int min, int max) {
  std::srand(seed != 0 ? seed : (unsigned int)time(NULL));
  seed = std::rand();
  std::default_random_engine generate(seed);
  std::uniform_int_distribution<int> distribution(min, max);
  int random_number = distribution(generate);
  return random_number;
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