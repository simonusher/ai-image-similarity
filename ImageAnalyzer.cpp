#include <utility>

#include <utility>

//
// Created by Szymon on 27.05.2019.
//

#include "ImageAnalyzer.h"

ImageAnalyzer::ImageAnalyzer(int neighbourhoodSize,
                             double cohesionThreshold,
                             vector<KeyPoint *> firstImageKeyPoints,
                             vector<KeyPoint *> secondImageKeyPoints) : neighbourhoodSize(neighbourhoodSize),
                                                                        cohesionThreshold(cohesionThreshold),
                                                                        firstImageKeyPoints(std::move(firstImageKeyPoints)),
                                                                        secondImageKeyPoints(std::move(
                                                                                secondImageKeyPoints)) {}

void ImageAnalyzer::analyze() {
    calculatePairs();
    calculateNeighbourhoods();
    analyzeNeigbourhoodCohesion();
}

void ImageAnalyzer::calculatePairs() {
    this->keyPointPairs = KeyPoint::getKeyPointPairs(this->firstImageKeyPoints, this->secondImageKeyPoints);
    this->firstImagePairedKeyPoints.reserve(keyPointPairs.size());
    this->secondImagePairedKeyPoints.reserve(keyPointPairs.size());
    for (int i = 0; i < keyPointPairs.size(); ++i) {
        firstImagePairedKeyPoints.push_back(keyPointPairs[i].first);
        secondImagePairedKeyPoints.push_back(keyPointPairs[i].second);
    }
}

void ImageAnalyzer::calculateNeighbourhoods() {
    for(KeyPoint* keyPoint : firstImagePairedKeyPoints){
        keyPoint->calculateNeighbourhood(this->firstImagePairedKeyPoints, this->neighbourhoodSize);
    }
    for(KeyPoint* keyPoint : secondImagePairedKeyPoints){
        keyPoint->calculateNeighbourhood(this->secondImagePairedKeyPoints, this->neighbourhoodSize);
    }
}

void ImageAnalyzer::analyzeNeigbourhoodCohesion() {
    for(const auto& keyPointPair : keyPointPairs){
        int neighbourPairsNumber = 0;
        for(int i = 0; neighbourPairsNumber < neighbourhoodSize && i < keyPointPairs.size(); i++){
            pair<KeyPoint*, KeyPoint*>& otherPair = keyPointPairs[i];
            if(keyPointPair.first != otherPair.first && keyPointPair.second != otherPair.second){
                if(keyPointPair.first->neighbourhoodContains(otherPair.first) && keyPointPair.second->neighbourhoodContains(otherPair.second)){
                    neighbourPairsNumber++;
                }
            }
        }
        if((neighbourPairsNumber / ((double)neighbourhoodSize)) > cohesionThreshold){
            coherentKeyPointPairs.push_back(keyPointPair);
        }
    }
}
