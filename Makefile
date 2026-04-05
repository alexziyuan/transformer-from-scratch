build-cpp:
	cd cpp && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(shell nproc)

test-cpp:
	./cpp/build/test_ops

bench:
	./cpp/build/bench_matmul
