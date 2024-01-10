# nvcc main.cpp -I./include -lcuda -o main.out
hipcc main.cpp -I./include -o main.out
# hipcc main.cpp -I./include -o main.out -gdwarf-4 -O0 -D_GLIBCXX_DEBUG

# hipcc test_one_model.cpp -I./include -o test_one_model.out
# hipcc test_one_kernel.cpp -I./include -o test_one_kernel.out
# hipcc test_priority_service.cpp -I./include -o test_priority_service.out
# hipcc test_right_size.cpp -I./include -o test_right_size.out
# hipcc compare_service.cpp -I./include -o compare_service.out
# hipcc test_co_kernels.cpp -I./include -o test_co_kernels.out
