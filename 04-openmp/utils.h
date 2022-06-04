#ifndef _FUNCOES_UTEIS
#define _FUNCOES_UTEIS

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

#define WMAT 2
#define WMIS (-1)
#define WGAP (-1)
#define null NULL

typedef struct Node Node;

struct Node {
  int value;
  int i;
  int j;
  Node *previous;
};

typedef struct {
  int length;
  std::string value;
} Sequence;

void get_random_number();
int random_integer(int min, int max);

int compare(char a, char b);
int min(int a, int b);
int max(int a, int b);
int maximo(int a, int b, int c);

void copy_str(char *dest, char *src, int start, int length);

void show_sequence(char sequence[], int sequence_length);
void show_result(std::vector<char> s1, std::vector<char> s2,
                 std::vector<char> resultado);
// void generate_subsequences(std::vector<Sequence> *destination, std::string
// sequence);

// int calcula_busca_local(const std::string sa, const std::string sb);
int simple_score(std::string s1, std::string s2);

#endif
