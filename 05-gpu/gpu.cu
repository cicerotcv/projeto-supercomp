#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <omp.h>

#include <iostream>

#include "utils.h"

typedef struct {
  int score;
  std::string seq1;
  std::string seq2;
} Result;

typedef struct {
  int start;
  int length;
} Subsequence;

// https://stackoverflow.com/a/1088299
// << operator to "print" vectors as string sequence
std::ostream& operator<<(std::ostream& os, const thrust::device_vector<char> vector) {
  for (auto el : vector) {
    os << el;
  }
  return os;
}

std::string get_subsequence(std::string String, int start, int length) {
  return String.substr(start, length);
}

struct Compute {
  thrust::device_ptr<char> dS;
  thrust::device_ptr<int> previous_row;
  char chr;

  Compute(thrust::device_ptr<char> _dS, char _chr,  thrust::device_ptr<int> _previous_row)
      : dS(_dS), chr(_chr), previous_row(_previous_row){};

  __host__ __device__ int operator()(const int(&i)) {
    // ...(i-1,j-1) ( i , j-1) (i+1,j-1)
    // ...(i-1, j ) ( i ,  j ) (i+1, j )

    int score;
    char comparing_char = dS[i];

    if (chr == comparing_char)
      score = previous_row[i - 1] + WMAT;
    else if (chr == '-' || comparing_char == '-')
      score = previous_row[i - 1] + WGAP;
    else
      score = previous_row[i - 1] + WMIS;

    return score > 0 ? score : 0;
  }
};

int subsequences_score(const std::string ssA, const std::string ssB){
  const int N = ssA.size();
  const int M = ssB.size();

  // std::cout << ssA << " x " << ssB << std::endl;

  thrust::device_vector<int> previous_row(N + 1);
  thrust::device_vector<int> current_row(N + 1);

  previous_row.resize(N + 1);
  current_row.resize(N + 1);

  thrust::fill(previous_row.begin(), previous_row.end(), 0);

  thrust::device_vector<char> dS(N);
  thrust::copy(ssA.begin(), ssA.begin() + N, dS.begin());

  thrust::counting_iterator<int> c0(1);
  thrust::counting_iterator<int> c1(M + 1);

  for (int i = 0; i < M; i++) {
    char comparing_char = ssB[i];
    thrust::transform(c0, c1, current_row.begin() + 1, Compute(dS.data(), comparing_char, previous_row.data()));
    thrust::inclusive_scan(current_row.begin() + 1, current_row.end(), previous_row.begin() + 1, thrust::maximum<int>());
  }

  return thrust::reduce(current_row.begin() + 1, current_row.end(), -1, thrust::maximum<int>());
}


void run() {
  int N, M;

  std::cin >> N;
  std::cin >> M;
  std::cout << std::endl;
  std::cout << "N: " << N << ", M: " << M << std::endl;

  double before, after;


  std::string A;
  std::string B;

  std::cin >> A;
  std::cin >> B;

  std::cout << "A: " << A << std::endl;
  std::cout << "B: " << B << std::endl;

  std::vector<std::string> SA;
  std::vector<std::string> SB;

  SA.reserve((N * (N + 1)) / 2);
  SB.reserve((M * (M + 1)) / 2);

  #pragma omp parallel // 1e-05 s
  {
    #pragma omp master
    {
      before = omp_get_wtime();

      #pragma omp task shared(SA)
      for (int length = 1; length <= N; length++) {
        for (int start = 0; start < N - length + 1; start++) {
          SA.push_back(get_subsequence(A, start, length));
        }
      }

      #pragma omp task shared(SB)
      for (int length = 1; length <= M; length++) {
        for (int start = 0; start < M - length + 1; start++) {
          SB.push_back(get_subsequence(B, start, length));
        }
      }
      
      #pragma omp taskwait
      {
        after = omp_get_wtime();
        std::cout << "Time to generate all subsequences: " << after - before << " s" <<std::endl;
      }
    }
  } 

  int max_score = 0;

  #pragma omp parallel shared(max_score)
  {
    #pragma omp for reduction(max : max_score)
    for (int index = 0; index < SA.size() * SB.size(); index++) {
      // A B C D  -> 4
      // E F G    -> 3  -> 12 combinações
      // AE   AF   AG   BE   BF   BG   CE   CF   CF   DE   DF   DG
      // 0x0  0x1  0x2  1x0  1x1  1x2  2x0  2x1  2x2  3x0  3x1  3x2
      // 0    1    2    3    4    5    6    7    8    9    10   11
      // (index // 4) x (index % 3)
      int indexA = (int) index / SA.size();
      int indexB = (int) index % SB.size();

      std::string ssA = SA.at(indexA);
      std::string ssB = SB.at(indexB);

      int local_score = subsequences_score(ssA, ssB);

      if (local_score > max_score) {
        max_score = local_score;
      }
    }
  }

  std::cout << "Max Score: " << max_score << std::endl;
}

int main() {
  double before = omp_get_wtime();
  run();
  double after = omp_get_wtime();

  std::cout << "Elapsed time: " <<  after - before << std::endl;
  return 0;
}