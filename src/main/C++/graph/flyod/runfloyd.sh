rm out
nvcc floyd.cu ../../matlib/matrix.cpp -o out
echo "Compilation successful, running matrix multiplication\n"
./out
