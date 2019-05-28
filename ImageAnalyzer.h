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

enum RansacHeuristic {
    Distance,
    Distribution,
    Iterations
};

enum TransformationType {
    Affine,
    Perspective
};

class ImageAnalyzer {
public:
    ImageAnalyzer(int neighbourhoodSize,
                  double cohesionThreshold,
                  int ransacIterations,
                  double transformationErrorThreshold,
                  string& firstImagePath,
                  string& secondImagePath);

    ~ImageAnalyzer();
    void init();
    void analyze();
    void runDemonstration();
    static void extractFeatures(const string& filePath);

    static const string FEATURE_DATA_FILE_SUFFIX;

private:
    void runRansacImpl();
    void showAllPairs();
    void showCoherentPairs();
    void showPairs(vector<pair<KeyPoint *, KeyPoint *>>& pairs, const string& windowName);
    void showPairsMatchingTransform();
    void calculatePairs();
    void calculateNeighbourhoods();
    void analyzeNeigbourhoodCohesion();
    Eigen::MatrixXd nextRandomAffineTransform();
    Eigen::MatrixXd nextRandomPerspectiveTransform();

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
    Eigen::MatrixXd bestFoundTransformation;

    std::default_random_engine randomEngine;
    int neighbourhoodSize;

    double cohesionThreshold;

    void runRansac();
};


#endif //INC_4_IMAGE_SIMILARITY_C_IMAGEANALYZER_H
