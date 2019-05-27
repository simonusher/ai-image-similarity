//
// Created by Szymon on 27.05.2019.
//
#include "KeyPoint.h"
#include <cstdlib>
#include <opencv2/opencv.hpp>
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
            vector<KeyPoint *> firstImageKeyPoints,
            vector<KeyPoint *> secondImageKeyPoints);
    void init();
    void analyze();
    void runDemonstration();
    static void extractFeatures(const string& filePath);

    static const string FEATURE_DATA_FILE_SUFFIX;

    vector<pair<KeyPoint*, KeyPoint*>> coherentKeyPointPairs;
private:
    void showAllPairs();
    void showCoherentPairs();
    void showPairs(vector<pair<KeyPoint *, KeyPoint *>>& pairs, const string& windowName, bool hstack = true);
    bool initialized;
    void calculatePairs();
    void calculateNeighbourhoods();

    void analyzeNeigbourhoodCohesion();
    string firstImagePath;
    string secondImagePath;
    vector<KeyPoint*> firstImageKeyPoints;

    vector<KeyPoint*> secondImageKeyPoints;
    vector<KeyPoint*> firstImagePairedKeyPoints;

    vector<KeyPoint*> secondImagePairedKeyPoints;
    vector<pair<KeyPoint*, KeyPoint*>> keyPointPairs;

    int neighbourhoodSize;
    double cohesionThreshold;
};


#endif //INC_4_IMAGE_SIMILARITY_C_IMAGEANALYZER_H
