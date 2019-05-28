#include <iostream>
#include "KeyPoint.h"
#include "chrono"
#include "ImageAnalyzer.h"
#include <boost/program_options.hpp>
using namespace std;
namespace po = boost::program_options;

bool fileExists(const string& fileName){
    return ifstream(fileName).good();
}

int main(int argc, char *argv[]) {
    int heuristicType;
    int ransacIterations;
    int neighbourhoodSize;
    double cohesionThreshold;
    double transformationThreshold;
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("images", po::value<std::vector<std::string>>(), "input images file names[2]")
            ("neighbourhood,n", po::value<int>(&neighbourhoodSize)->default_value(150), "neighbourhood size for cohesion analysis, default=150")
            ("cohesion,c", po::value<double>(&cohesionThreshold)->default_value(0.6), "cohesion threshold for analysis, default=0.6")
            ("iter,i", po::value<int>(&ransacIterations)->default_value(10000), "ransac iteration number, default=10000")
            ("transformation-threshold,t", po::value<double>(&transformationThreshold)->default_value(1), "transformation threshold for ransac default=1")
            ("perspective,p", "use perspective transformation instead of default affine")
            ("heuristic,h", po::value<int>(&heuristicType)->default_value(0), "choose ransac heuristic")
            ;
    po::positional_options_description posDesc;
    posDesc.add("images", -1);
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
            options(desc).positional(posDesc).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }
    else {
        vector<string> images = vm["images"].as<vector<string>>();
        if(images.size() != 2){
            std::cout << "Please specify exactly 2 image file names!";
            return -1;
        }
        else {
            string firstFileName(argv[1]);
            string secondFileName(argv[2]);
            if(!fileExists(firstFileName)){
                std::cout << "File: " << firstFileName << " doesn't exist!";
                return -1;
            } else if(!fileExists(secondFileName)){
                std::cout << "File: " << secondFileName << " doesn't exist!";
                return -1;
            } else {
                TransformationType transformationType = Affine;
                if(vm.count("perspective")){
                    transformationType = Perspective;
                }
                auto start = chrono::system_clock::now();
                ImageAnalyzer imageAnalyzer(neighbourhoodSize, cohesionThreshold, ransacIterations,
                        transformationThreshold, transformationType, firstFileName, secondFileName);
                imageAnalyzer.analyze();
                auto end = chrono::system_clock::now();
                std::cout << "Elapsed time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " milis." << std:: endl;
                imageAnalyzer.runDemonstration();
                return 0;
            }
        }
    }
}