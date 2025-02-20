OpenCv verion: 4.11.0

1. Compile the shared library

clang++ -shared -o libopencv_wrapper.so -fPIC -std=c++11 opencv_dnn_wrapper.cpp `pkg-config --cflags --libs opencv4`

2. Compile the main program
gcc-14 -o main main.c input_data.c -L. -lopencv_wrapper -Wl,-rpath,.

3. Run the program
export DYLD_LIBRARY_PATH=.:$DYLD_LIBRARY_PATH  # macOS
./main

clang++ -shared -o libopencv_wrapper.so -fPIC -std=c++11 opencv_dnn_wrapper.cpp `pkg-config --cflags --libs opencv4`
gcc-14 -o main main.c input_data.c -L. -lopencv_wrapper -Wl,-rpath,.
./main