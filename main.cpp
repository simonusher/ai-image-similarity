#include <iostream>
#include "KeyPoint.h"
#include "chrono"
#include "ImageAnalyzer.h"
using namespace std;
int main(int argc, char *argv[]) {
    if(argc < 3){
        std::cout << "Provide two file names as arguments!";
    }
    else {
        string firstFileName(argv[1]);
        string secondFileName(argv[2]);
        if(!ifstream(firstFileName).good()){
            std::cout << "File: " << firstFileName << " doesn't exist!";
            return -1;
        } else if(!ifstream(secondFileName).good()){
            std::cout << "File: " << secondFileName << " doesn't exist!";
            return -1;
        } else {
            auto start = chrono::system_clock::now();
            ImageAnalyzer imageAnalyzer(150, 0.8, 10000, 1, firstFileName, secondFileName);
            imageAnalyzer.analyze();
            auto end = chrono::system_clock::now();
            std::cout << "Elapsed time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " milis." << std:: endl;
            imageAnalyzer.runDemonstration();
            return 0;
        }
    }
}