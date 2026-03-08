# Bio-AI-Project-2
A simple C++ project.

## Build

### Windows (Visual Studio/MSVC)
```bash
mkdir build
cd build
cmake ..
cd ..
cmake --build build
.\build\Debug\main.exe
```
Alternatively, there is also a build script included that builds and runs the project:
```bash
mkdir build
./build.bat
```

### Linux/macOS (GCC/Clang)
```bash
mkdir build && cd build
cmake ..
make
./main
```

## Running different parameters
The project is set up to manually enter parameters and settings needed to reproduce results. 
The runRuinAndRepairGA method in main.cpp contains the setup of the solutions. 
Enter the name of the dataset as a parameter to HomeCare, and then enter the desired parameters.
The output is written to a file in the working directory with a timestamp as the name. 
