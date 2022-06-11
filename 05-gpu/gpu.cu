#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>

#include "utils.h"

typedef struct {
  int score;
  std::string seq1;
  std::string seq2;
} Result;

// https://stackoverflow.com/a/1088299
// << operator to "print" vectors as string sequence
std::ostream& operator<<(std::ostream& os, const thrust::device_vector<char> vector) {
  for (int i = 0; i < vector.size(); i++) {
    os << vector[i];
  }
  return os;
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

// struct subsequence {
//   typedef std::string::iterator iterator;
//   __host__ __device__ int operator()(const iterator &begin , const iterator&
//   end) { return 0; }
// };

// void generate_subsequences(std::vector<Sequence>* destination, std::string
// sequence) {
//   const int N = sequence.length();

//   for (int length = 1; length <= N; length++) {
//     for (int pos = 0; pos <= N - length; pos++) {
//       std::string value(sequence.substr(pos, length));
//       Sequence sequence = {length, value};
//       destination->push_back(sequence);
//     }
//   }
// }
// void generate_subsequences(std::vector<Sequence>* destination, std::string
// sequence) {
//   const int N = sequence.length();
//   for (int length = 1; length <= N; length++) {
//     for (int pos = 0; pos <= N - length; pos++) {
//       std::string value(sequence.substr(pos, length));
//       Sequence sequence = {length, value};
//       destination->push_back(sequence);
//     }
//   }
// }

thrust::device_vector<char> get_subsequence(
    const thrust::device_vector<char>& sequence, int start, int size) {
  return thrust::device_vector<char>(sequence.begin() + start,
                                     sequence.begin() + start + size);
}

// int calculate_score(const thrust::device_vector<char> &sequence1, const
// thrust::device_vector<char> &sequence2) {
//   int max = std::max(sequence1.size(), sequence2.size());

//   thrust::device_vector<thrust::device_vector<char>> kernel;

//   thrust::counting_iterator<int> i(1);
//   thrust::counting_iterator<int> j(max + 1);

// }

struct S_temp {
  const thrust::device_vector<char> SN;
  const thrust::device_vector<char> SM;

  // construtor
  S_temp(const thrust::device_vector<char> _SN,
         const thrust::device_vector<char> _SM)
      : SN(_SN), SM(_SM) {}

  __host__ __device__ int operator()(const int i, const int j) {
    char a, b;
    a = SN[i];
    b = SM[j];
    if (a == b) return WMAT;
    if (a == '-' || b == '-') return WGAP;
    return WMIS;
  }
};

void run() {
  int N, M;

  std::cin >> N;
  std::cin >> M;
  std::cout << std::endl;
  std::cout << "N: " << N << ", M: " << M << std::endl;

  std::string a;
  std::string b;

  std::cin >> a;
  std::cin >> b;

  thrust::device_vector<char> gpu_a(a.begin(), a.end());  // string::sequencia 1
  thrust::device_vector<char> gpu_b(b.begin(), b.end());  // string::sequencia 2
  thrust::device_vector<char> score;                      // int::score

  std::cout << gpu_a << std::endl;
  std::cout << gpu_b << std::endl;

  // S_temp s_temp(gpu_a, gpu_b);

  // thrust::device_vector<int> previous_row;
  // thrust::fill(previous_row.begin(), previous_row.end(), 0);

  // thrust::device_vector<int> current_row;

  // thrust::transform(gpu_a.begin(), gpu_a.end(), gpu_b.begin(), gpu_b.end(),
  // score.begin(), compare);

  // thrust::transform(current_row.begin(), current_row.back(),
  // previous_row.begin(), previous_row.end(), s_temp());

  // for (int i = 0; i < N; i++) {
  // kernel.push_back(row);
  // }

  // std::cout << kernel.size() << std::endl;

  // thrust::device_vector<char> my_string = get_subsequence(gpu_a, 0, 2);

  // for (char chr : my_string) {
  //   std::cout << chr << std::endl;
  // }

  // for (int i = 0; i < gpu)

  // thrust::device_vector<Sequence> sn;
  // thrust::device_vector<Sequence> sm;

  // // gera todas as subsequências de tamanho 1 até N
  // generate_subsequences(&sn, a);

  // // gera todas as subsequências de tamanho 1 até M
  // generate_subsequences(&sm, b);

  // Result first_method = {0};
  // Result second_method = {0};

  // // // Heurística de Alinhamento Local
  // calculate_score(sm, sn, calcula_busca_local, &first_method);

  // // Comparação Simples
  // calculate_score(sm, sn, same_size, &second_method);

  // Result* result;
  // if (first_method.score > second_method.score) {
  //   result = &first_method;
  // } else {
  //   result = &second_method;
  // }

  // std::cout << "Max score: " << result->score << std::endl;
}

int main() {
  clock_t before = std::clock();
  run();
  clock_t after = std::clock();

  double delta_time = (double)(after - before) / CLOCKS_PER_SEC;

  std::cout << "Elapsed time: " << delta_time << std::endl;
  return 0;
}