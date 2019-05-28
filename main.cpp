#include <iostream>
#include "KeyPoint.h"
#include "chrono"
#include "ImageAnalyzer.h"
using namespace std;
int main() {
    auto start = chrono::system_clock::now();
    string firstFileName("1.png");
    string secondFileName("2.png");
    ImageAnalyzer imageAnalyzer(100, 0.6, 100000, 15, firstFileName, secondFileName);
    imageAnalyzer.analyze();
    auto end = chrono::system_clock::now();
    std::cout << "time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << std:: endl;
    imageAnalyzer.runDemonstration();
    return 0;
}