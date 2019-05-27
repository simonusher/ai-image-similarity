#include <utility>

#include <utility>

//
// Created by Szymon on 27.05.2019.
//

#include "ImageAnalyzer.h"

const string ImageAnalyzer::FEATURE_DATA_FILE_SUFFIX = ".haraff.sift";

ImageAnalyzer::ImageAnalyzer(int neighbourhoodSize,
                             double cohesionThreshold,
                             vector<KeyPoint *> firstImageKeyPoints,
                             vector<KeyPoint *> secondImageKeyPoints) : initialized(true),
                                                                        neighbourhoodSize(neighbourhoodSize),
                                                                        cohesionThreshold(cohesionThreshold),
                                                                        firstImageKeyPoints(std::move(firstImageKeyPoints)),
                                                                        secondImageKeyPoints(std::move(
                                                                                secondImageKeyPoints)) {}

void ImageAnalyzer::analyze() {
    if(!initialized){
        init();
    }
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
        if((neighbourPairsNumber / ((double)neighbourhoodSize)) >= cohesionThreshold){
            coherentKeyPointPairs.push_back(keyPointPair);
        }
    }
}

void ImageAnalyzer::extractFeatures(const string &filePath) {
    string command = "extract_features -haraff -sift -i " + filePath;
    system(command.c_str());
}

ImageAnalyzer::ImageAnalyzer(int neighbourhoodSize, double cohesionThreshold, string &firstImagePath,
                             string &secondImagePath) : firstImagePath(firstImagePath),
                                                        secondImagePath(secondImagePath),
                                                        initialized(false),
                                                        neighbourhoodSize(neighbourhoodSize),
                                                        cohesionThreshold(cohesionThreshold) {}

void ImageAnalyzer::init() {
    extractFeatures(firstImagePath);
    extractFeatures(secondImagePath);
    firstImageKeyPoints = KeyPoint::importKeyPoints(firstImagePath + FEATURE_DATA_FILE_SUFFIX);
    secondImageKeyPoints = KeyPoint::importKeyPoints(secondImagePath + FEATURE_DATA_FILE_SUFFIX);
}

void ImageAnalyzer::runDemonstration() {
    showAllPairs();
    showCoherentPairs();
}

void ImageAnalyzer::showCoherentPairs() {
    showPairs(coherentKeyPointPairs, "Coherent key point pairs");
}

void ImageAnalyzer::showAllPairs() {
    showPairs(keyPointPairs, "All key point pairs");
}

void ImageAnalyzer::showPairs(vector<pair<KeyPoint *, KeyPoint *>> &pairs, const string& windowName, bool hstack) {
    cv::Mat im1 = cv::imread(firstImagePath, 1);
    cv::Mat im2 = cv::imread(secondImagePath, 1);
    cv::Mat imstack;
    int vOffset = 0;
    int hOffset = 0;
    if(hstack) {
        cv::hconcat(im1, im2, imstack);
        hOffset = im1.cols;
    } else {
        cv::vconcat(im1, im2, imstack);
        vOffset = im1.rows;
    }
    for(auto& keyPointPair : pairs) {
        cv::Point firstPoint(keyPointPair.first->getX(), keyPointPair.first->getY());
        cv::Point secondPoint(keyPointPair.second->getX() + hOffset, keyPointPair.second->getY() + vOffset);
        cv::line(imstack, firstPoint, secondPoint, cv::Scalar(0, 0, 255));
    }
    namedWindow(windowName, cv::WINDOW_AUTOSIZE );
    imshow(windowName, imstack);
    cv::waitKey(0);
}
