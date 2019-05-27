//
// Created by Szymon on 27.05.2019.
//
#include "KeyPoint.h"
#ifndef INC_4_IMAGE_SIMILARITY_C_IMAGEANALYZER_H
#define INC_4_IMAGE_SIMILARITY_C_IMAGEANALYZER_H


class ImageAnalyzer {
public:
    ImageAnalyzer(int neighbourhoodSize,
            double cohesionThreshold,
            vector<KeyPoint *> firstImageKeyPoints,
            vector<KeyPoint *> secondImageKeyPoints);
    void analyze();
    vector<pair<KeyPoint*, KeyPoint*>> coherentKeyPointPairs;
private:
    void calculatePairs();
    void calculateNeighbourhoods();

    void analyzeNeigbourhoodCohesion();
    vector<KeyPoint*> firstImageKeyPoints;

    vector<KeyPoint*> secondImageKeyPoints;
    vector<KeyPoint*> firstImagePairedKeyPoints;

    vector<KeyPoint*> secondImagePairedKeyPoints;
    vector<pair<KeyPoint*, KeyPoint*>> keyPointPairs;

    int neighbourhoodSize;
    double cohesionThreshold;
};


#endif //INC_4_IMAGE_SIMILARITY_C_IMAGEANALYZER_H
