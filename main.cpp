#include <iostream>
#include "KeyPoint.h"
#include "chrono"
#include "ImageAnalyzer.h"
using namespace std;
int main() {
    string firstFileName("1.png");
    string secondFileName("2.png");
    ImageAnalyzer imageAnalyzer(100, 0.9, firstFileName, secondFileName);
    imageAnalyzer.analyze();
    imageAnalyzer.runDemonstration();
    return 0;
}