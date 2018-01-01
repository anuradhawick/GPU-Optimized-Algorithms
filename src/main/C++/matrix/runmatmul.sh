rm out
nvcc matMul.cu ../matlib/matrix.cpp -o out
echo "Compilation successful, running matrix multiplication\n"
./out
