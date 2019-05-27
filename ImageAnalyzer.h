//
// Created by Szymon on 27.05.2019.
//
#include "KeyPoint.h"
#include <cstdlib>
#include <utility>
#include <random>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
using std::unordered_set;
#ifndef INC_4_IMAGE_SIMILARITY_C_IMAGEANALYZER_H
#define INC_4_IMAGE_SIMILARITY_C_IMAGEANALYZER_H

class ImageAnalyzer {
public:
    ImageAnalyzer(int neighbourhoodSize,
            double cohesionThreshold,
            string& firstImagePath,
            string& secondImagePath);

    ImageAnalyzer(int neighbourhoodSize,
                  double cohesionThreshold,
                  int ransacIterations,
                  double transformationErrorThreshold,
                  string& firstImagePath,
                  string& secondImagePath);

    ImageAnalyzer(int neighbourhoodSize,
            double cohesionThreshold,
            vector<KeyPoint *> firstImageKeyPoints,
            vector<KeyPoint *> secondImageKeyPoints);

    ~ImageAnalyzer();
    void init();
    void analyze();
    void runDemonstration();
    static void extractFeatures(const string& filePath);

    static const string FEATURE_DATA_FILE_SUFFIX;

private:
    void runRansacAffine();
    void showAllPairs();
    void showCoherentPairs();
    void showPairs(vector<pair<KeyPoint *, KeyPoint *>>& pairs, const string& windowName, bool hstack = true);
    void showPairsMatchingTransform();
    void calculatePairs();
    void calculateNeighbourhoods();
    void analyzeNeigbourhoodCohesion();
    Eigen::MatrixXd nextRandomAffineTransform();

    vector<pair<KeyPoint*, KeyPoint*>> getNDifferentCoherentKeyPointPairs(int n);
    bool initialized;
    int ransacIterations;
    double transformationErrorThreshold;
    string firstImagePath;
    string secondImagePath;
    vector<KeyPoint*> firstImageKeyPoints;
    vector<KeyPoint*> secondImageKeyPoints;

    vector<KeyPoint*> firstImagePairedKeyPoints;
    vector<KeyPoint*> secondImagePairedKeyPoints;

    vector<pair<KeyPoint*, KeyPoint*>> keyPointPairs;
    vector<pair<KeyPoint*, KeyPoint*>> coherentKeyPointPairs;
    vector<pair<KeyPoint*, KeyPoint*>> matchingTransformKeyPointPairs;
    Eigen::Matrix3d bestFoundTransformation;

    std::default_random_engine randomEngine;
    int neighbourhoodSize;

    double cohesionThreshold;

    void runRansac();
};


#endif //INC_4_IMAGE_SIMILARITY_C_IMAGEANALYZER_H
