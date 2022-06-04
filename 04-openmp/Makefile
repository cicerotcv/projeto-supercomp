IN_PARALLEL=main_parallel
IN_SEQUENTIAL=main_sequential

OUT_SEQUENTIAL=sequential
OUT_PARALLEL=parallel


all: parallel sequential

clear: clear_parallel clear_sequential

clear_sequential:
	rm -f $(OUT_SEQUENTIAL);

clear_parallel:
	rm -f $(OUT_PARALLEL);

parallel: clear_parallel
	g++ -Wall -O3 -fopenmp utils.cpp $(IN_PARALLEL).cpp -o $(OUT_PARALLEL)

sequential: clear_sequential
	g++ -Wall -O3 -fopenmp utils.cpp $(IN_SEQUENTIAL).cpp -o $(OUT_SEQUENTIAL)
