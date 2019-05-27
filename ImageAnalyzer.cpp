
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

ImageAnalyzer::ImageAnalyzer(int neighbourhoodSize, double cohesionThreshold, int ransacIterations,
                             double transformationErrorThreshold, string &firstImagePath,string &secondImagePath) :
                                    ransacIterations(ransacIterations),
                                    transformationErrorThreshold(transformationErrorThreshold),
                                    firstImagePath(firstImagePath),
                                    secondImagePath(secondImagePath),
                                    initialized(false),
                                    neighbourhoodSize(neighbourhoodSize),
                                    cohesionThreshold(cohesionThreshold) {}

ImageAnalyzer::ImageAnalyzer(int neighbourhoodSize, double cohesionThreshold, string &firstImagePath,
                             string &secondImagePath) : firstImagePath(firstImagePath),
                                                        secondImagePath(secondImagePath),
                                                        initialized(false),
                                                        neighbourhoodSize(neighbourhoodSize),
                                                        cohesionThreshold(cohesionThreshold) {}

ImageAnalyzer::~ImageAnalyzer() {
    for(KeyPoint* keyPoint : firstImageKeyPoints){
        delete keyPoint;
    }
    for(KeyPoint* keyPoint : secondImageKeyPoints){
        delete keyPoint;
    }
}

void ImageAnalyzer::analyze() {
    if(!initialized){
        init();
    }
    calculatePairs();
    calculateNeighbourhoods();
    analyzeNeigbourhoodCohesion();
    runRansac();
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

void ImageAnalyzer::init() {
    std::random_device randomDevice;
    randomEngine = std::default_random_engine(randomDevice());
    extractFeatures(firstImagePath);
    extractFeatures(secondImagePath);
    firstImageKeyPoints = KeyPoint::importKeyPoints(firstImagePath + FEATURE_DATA_FILE_SUFFIX);
    secondImageKeyPoints = KeyPoint::importKeyPoints(secondImagePath + FEATURE_DATA_FILE_SUFFIX);
}

void ImageAnalyzer::runDemonstration() {
    showAllPairs();
    showCoherentPairs();
    showPairsMatchingTransform();
}

void ImageAnalyzer::showCoherentPairs() {
    showPairs(coherentKeyPointPairs, "Coherent key point pairs");
}

void ImageAnalyzer::showAllPairs() {
    showPairs(keyPointPairs, "All key point pairs");
}

void ImageAnalyzer::showPairsMatchingTransform() {
    showPairs(matchingTransformKeyPointPairs, "Matching transform key point pairs");
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
    std::uniform_int_distribution<int> colorDistribution(0, 255);
    for(auto& keyPointPair : pairs) {
        cv::Point firstPoint(keyPointPair.first->getX(), keyPointPair.first->getY());
        cv::Point secondPoint(keyPointPair.second->getX() + hOffset, keyPointPair.second->getY() + vOffset);
        cv::line(imstack, firstPoint, secondPoint,
                cv::Scalar(colorDistribution(randomEngine), colorDistribution(randomEngine), colorDistribution(randomEngine)));
    }
    namedWindow(windowName, cv::WINDOW_AUTOSIZE );
    imshow(windowName, imstack);
    cv::waitKey(0);
}

vector<pair<KeyPoint*, KeyPoint*>> ImageAnalyzer::getNDifferentCoherentKeyPointPairs(int n) {
    unordered_set<int> indices;
    std::uniform_int_distribution<int> distribution(0, coherentKeyPointPairs.size() -1);
    while(indices.size() != n) {
        int randomIndex = distribution(randomEngine);
        indices.insert(randomIndex);
    }
    vector<pair<KeyPoint*, KeyPoint*>> result;
    for(int index : indices){
        result.push_back(coherentKeyPointPairs[index]);
    }
    return result;
}

void ImageAnalyzer::runRansac() {
    runRansacAffine();
}

void ImageAnalyzer::runRansacAffine() {
    Eigen::Matrix3d bestTransformation;
    vector<pair<KeyPoint*, KeyPoint*>> bestConsensus;
    int bestScore = 0;
    for(int i = 0; i < ransacIterations; i++){
//        Eigen::MatrixXd A = nextRandomAffineTransform();
        Eigen::MatrixXd A = nextRandomPerspectiveTransform();
        vector<pair<KeyPoint*, KeyPoint*>> consensus;
        consensus.reserve(coherentKeyPointPairs.size());
        for(auto& pair : coherentKeyPointPairs){
            Eigen::Vector3d point;
            point << pair.first->getX(),
                    pair.first->getY(),
                    1;
            Eigen::Vector3d transformedPoint = A * point;
            double distance = pair.second->euclideanDistance(transformedPoint(0), transformedPoint(1));
            if(distance < transformationErrorThreshold){
                consensus.push_back(pair);
            }
        }
        if(consensus.size() > bestScore){
            bestScore = consensus.size();
            bestTransformation = A;
            bestConsensus = consensus;
            std::cout << "current score: " << bestScore << "best possible: " << coherentKeyPointPairs.size();
        }
    }
    this->bestFoundTransformation = bestTransformation;
    this->matchingTransformKeyPointPairs = bestConsensus;
}

Eigen::MatrixXd ImageAnalyzer::nextRandomAffineTransform() {
    vector<pair<KeyPoint*, KeyPoint*>> randomKeyPoints = getNDifferentCoherentKeyPointPairs(3);
    KeyPoint *firstPointA = randomKeyPoints[0].first;
    KeyPoint *secondPointA = randomKeyPoints[1].first;
    KeyPoint *thirdPointA = randomKeyPoints[2].first;
    KeyPoint *firstPointB = randomKeyPoints[0].second;
    KeyPoint *secondPointB = randomKeyPoints[1].second;
    KeyPoint *thirdPointB = randomKeyPoints[2].second;
    Eigen::MatrixXd X(6,6);
    X << firstPointA->getX(), firstPointA->getY(), 1, 0, 0, 0,
            secondPointA->getX(), secondPointA->getY(), 1, 0, 0, 0,
            thirdPointA->getX(), thirdPointA->getY(), 1, 0, 0, 0,
            0, 0, 0, firstPointA->getX(), firstPointA->getY(), 1,
            0, 0, 0, secondPointA->getX(), secondPointA->getY(), 1,
            0, 0, 0, thirdPointA->getX(), thirdPointA->getY(), 1;
    Eigen::VectorXd Y(6);
    Y << firstPointB->getX(),
            secondPointB->getX(),
            thirdPointB->getX(),
            firstPointB->getY(),
            secondPointB->getY(),
            thirdPointB->getY();
    Eigen::VectorXd vec = X.inverse() * Y;
    Eigen::MatrixXd result(3,3);
    result << vec(0), vec(1), vec(2),
            vec(3), vec(4), vec(5),
            0, 0, 1;
    return result;
}

Eigen::MatrixXd ImageAnalyzer::nextRandomPerspectiveTransform() {
    vector<pair<KeyPoint*, KeyPoint*>> randomKeyPoints = getNDifferentCoherentKeyPointPairs(4);
    KeyPoint *A1 = randomKeyPoints[0].first;
    KeyPoint *A2 = randomKeyPoints[1].first;
    KeyPoint *A3 = randomKeyPoints[2].first;
    KeyPoint *A4 = randomKeyPoints[3].first;
    KeyPoint *B1 = randomKeyPoints[0].second;
    KeyPoint *B2 = randomKeyPoints[1].second;
    KeyPoint *B3 = randomKeyPoints[2].second;
    KeyPoint *B4 = randomKeyPoints[3].second;
    Eigen::MatrixXd X(8,8);
    X << A1->getX(), A1->getY(), 1, 0, 0, 0, (-B1->getX() * A1->getX()), (-B1->getX() * A1->getY()),
         A2->getX(), A2->getY(), 1, 0, 0, 0, (-B2->getX() * A2->getX()), (-B2->getX() * A2->getY()),
         A3->getX(), A3->getY(), 1, 0, 0, 0, (-B3->getX() * A3->getX()), (-B3->getX() * A3->getY()),
         A4->getX(), A4->getY(), 1, 0, 0, 0, (-B4->getX() * A4->getX()), (-B4->getX() * A4->getY()),
         0, 0, 0, A1->getX(), A1->getY(), 1, (-B1->getY() * A1->getX()), (-B1->getY() * A1->getY()),
         0, 0, 0, A2->getX(), A2->getY(), 1, (-B2->getY() * A2->getX()), (-B2->getY() * A2->getY()),
         0, 0, 0, A3->getX(), A3->getY(), 1, (-B3->getY() * A3->getX()), (-B3->getY() * A3->getY()),
         0, 0, 0, A4->getX(), A4->getY(), 1, (-B4->getY() * A4->getX()), (-B4->getY() * A4->getY());

    Eigen::VectorXd Y(8);
    Y << B1->getX(),
            B2->getX(),
            B3->getX(),
            B4->getX(),
            B1->getY(),
            B2->getY(),
            B3->getY();
            B4->getY();
    Eigen::VectorXd vec = X.inverse() * Y;
    Eigen::MatrixXd result(3,3);
    result << vec(0), vec(1), vec(2),
            vec(3), vec(4), vec(5),
            vec(6), vec(7), 1;
    return result;
}
