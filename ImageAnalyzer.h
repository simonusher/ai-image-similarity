//
// Created by Szymon on 27.05.2019.
//
#include "KeyPoint.h"
#include <cstdlib>
#include <utility>
#include <random>
#include <chrono>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
using std::unordered_set;
#ifndef INC_4_IMAGE_SIMILARITY_C_IMAGEANALYZER_H
#define INC_4_IMAGE_SIMILARITY_C_IMAGEANALYZER_H

enum RansacHeuristic {
    None,
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
                  TransformationType transformationType,
                  RansacHeuristic ransacHeuristic,
                  string& firstImagePath,
                  string& secondImagePath,
                  bool showTransformedImage,
                  bool showTimes);

    ~ImageAnalyzer();
    void init();
    void analyze();
    void runDemonstration();
    static void extractFeatures(const string& filePath);

    static const string FEATURE_DATA_FILE_SUFFIX;

private:
    void runRansac();
    void runRansacImpl();
    void showAllPairs();
    void showCoherentPairs();
    void showPairs(vector<pair<KeyPoint *, KeyPoint *>>& pairs, const string& windowName);
    void showPairsMatchingTransform();
    void showTransformedImage();
    void calculatePairs();
    void calculateNeighbourhoods();
    void analyzeNeigbourhoodCohesion();
    Eigen::MatrixXd nextRandomAffineTransform();
    Eigen::MatrixXd nextRandomPerspectiveTransform();
    vector<pair<KeyPoint*, KeyPoint*>> getNDifferentKeyPointPairs(int n);
    vector<pair<KeyPoint*, KeyPoint*>> getNDifferentKeyPointPairsHeuristic(int n);
    vector<pair<KeyPoint*, KeyPoint*>> getNPairs(int n);
    
    bool distanceHeuristicCorrect(pair<KeyPoint*, KeyPoint*>& firstPair, pair<KeyPoint*, KeyPoint*>& secondPair);

    double smallRSquared;
    double bigRSquared;
    bool initialized;
    bool shouldShowTransformedImage;
    bool showTimes;
    long pairCalculationTime;
    long coherenceAnalysisTime;
    long ransacTime;

    int ransacIterations;
    int neighbourhoodSize;
    double cohesionThreshold;
    double transformationErrorThreshold;
    string firstImagePath;
    string secondImagePath;
    RansacHeuristic ransacHeuristic;
    TransformationType transformationType;

    vector<KeyPoint*> firstImageKeyPoints;
    vector<KeyPoint*> secondImageKeyPoints;

    vector<KeyPoint*> firstImagePairedKeyPoints;
    vector<KeyPoint*> secondImagePairedKeyPoints;

    vector<pair<KeyPoint*, KeyPoint*>> keyPointPairs;
    vector<pair<KeyPoint*, KeyPoint*>> coherentKeyPointPairs;
    vector<pair<KeyPoint*, KeyPoint*>> matchingTransformKeyPointPairs;
    Eigen::MatrixXd bestFoundTransformation;

    std::default_random_engine randomEngine;

    void initDistanceHeuristic();
};


#endif //INC_4_IMAGE_SIMILARITY_C_IMAGEANALYZER_H
