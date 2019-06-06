
//
// Created by Szymon on 27.05.2019.
//

#include "ImageAnalyzer.h"

const string ImageAnalyzer::FEATURE_DATA_FILE_SUFFIX = ".haraff.sift";

ImageAnalyzer::ImageAnalyzer(int neighbourhoodSize, double cohesionThreshold, int ransacIterations,
                             double transformationErrorThreshold, TransformationType transformationType,
                             RansacHeuristic ransacHeuristic, string &firstImagePath, string &secondImagePath,
                             bool showTransformedImage, bool showTimes, double ransacProbability) :
        ransacIterations(ransacIterations),
        transformationErrorThreshold(transformationErrorThreshold),
        firstImagePath(firstImagePath),
        secondImagePath(secondImagePath),
        initialized(false),
        neighbourhoodSize(neighbourhoodSize),
        cohesionThreshold(cohesionThreshold),
        transformationType(transformationType),
        shouldShowTransformedImage(showTransformedImage),
        showTimes(showTimes),
        ransacHeuristic(ransacHeuristic),
        ransacProbability(ransacProbability) {}


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

    std::cout << "Calculating key point pairs..." << std::endl;
    auto start = std::chrono::system_clock::now();
    calculatePairs();
    auto end = std::chrono::system_clock::now();
    pairCalculationTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Found " << keyPointPairs.size() << " key point pairs." << std::endl;
    if(showTimes){
        std::cout << "Pair calculation time: " << pairCalculationTime << "ms" << std::endl;
    }

    std::cout << "Calculating neighbourhoods..." << std::endl;
    start = std::chrono::system_clock::now();
    calculateNeighbourhoods();
    std::cout << "Analyzing cohesion..." << std::endl;
    analyzeNeigbourhoodCohesion();
    std::cout << "Found " << coherentKeyPointPairs.size() << " coherent pairs. "<< std::endl;
    end = std::chrono::system_clock::now();
    coherenceAnalysisTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if(showTimes){
        std::cout << "Coherence analysis time: " << coherenceAnalysisTime << "ms" << std::endl;
    }


    std::cout << "Running ransac" << std::endl;
    start = std::chrono::system_clock::now();
    runRansac();
    end = std::chrono::system_clock::now();
    ransacTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Found " << matchingTransformKeyPointPairs.size() << " pairs matching transform. " << std::endl;
    if(showTimes){
        std::cout << "Ransac time: " << ransacTime << "ms" << std::endl;
    }
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
    double thresholdValue = neighbourhoodSize * cohesionThreshold;
    for(const auto& keyPointPair : keyPointPairs) {
        int neighbourPairsNumber = 0;
        for(const auto& otherPair : keyPointPairs){
            if(keyPointPair != otherPair){
                if(keyPointPair.first->neighbourhoodContains(otherPair.first) && keyPointPair.second->neighbourhoodContains(otherPair.second)){
                    neighbourPairsNumber++;
                    if(neighbourPairsNumber > thresholdValue){
                        coherentKeyPointPairs.push_back(keyPointPair);
                        break;
                    }
                }
            }
        }
    }
}

void ImageAnalyzer::extractFeatures(const string &filePath) {
    if(!std::ifstream(filePath + FEATURE_DATA_FILE_SUFFIX).good()){
        string command = "extract_features -haraff -sift -i " + filePath;
        system(command.c_str());
    } else {
        std::cout << "Skipping file: " << filePath << ". Already extracted..." << std::endl;
    }
}

void ImageAnalyzer::init() {
    std::random_device randomDevice;
    randomEngine = std::default_random_engine(randomDevice());
    extractFeatures(firstImagePath);
    extractFeatures(secondImagePath);
    firstImageKeyPoints = KeyPoint::importKeyPoints(firstImagePath + FEATURE_DATA_FILE_SUFFIX);
    secondImageKeyPoints = KeyPoint::importKeyPoints(secondImagePath + FEATURE_DATA_FILE_SUFFIX);
    if(ransacHeuristic == Distance){
        initDistanceHeuristic();
    }
}

void ImageAnalyzer::runDemonstration() {
    showAllPairs();
    showCoherentPairs();
    showPairsMatchingTransform();
    if(shouldShowTransformedImage){
        showTransformedImage();
    }
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

void ImageAnalyzer::showTransformedImage() {
    cv::Mat warp_dst;
    if(transformationType == Affine){
        double m[2][3] = {{bestFoundTransformation(0,0), bestFoundTransformation(0,1), bestFoundTransformation(0,2)},
                          {bestFoundTransformation(1,0), bestFoundTransformation(1,1), bestFoundTransformation(1,2)}};
        cv::Mat warp_mat( 2, 3, CV_64FC1, m);
        cv::Mat src = cv::imread(firstImagePath, 1);
        warp_dst = cv::Mat::zeros( src.rows, src.cols, src.type() );
        cv::warpAffine(src, warp_dst, warp_mat, warp_dst.size());
    } else {
        double m[3][3] = {{bestFoundTransformation(0,0), bestFoundTransformation(0,1), bestFoundTransformation(0,2)},
                          {bestFoundTransformation(1,0), bestFoundTransformation(1,1), bestFoundTransformation(1,2)},
                          {bestFoundTransformation(2,0), bestFoundTransformation(2,1), bestFoundTransformation(2,2)}};
        cv::Mat warp_mat( 3, 3, CV_64FC1, m);
        cv::Mat src = cv::imread(firstImagePath, 1);
        warp_dst = cv::Mat::zeros( src.rows, src.cols, src.type() );
        cv::warpPerspective(src, warp_dst, warp_mat, warp_dst.size());
    }

    namedWindow( "Transformed", cv::WINDOW_AUTOSIZE );
    imshow( "Transformed", warp_dst );
    cv::waitKey(0);
}

void ImageAnalyzer::showPairs(vector<pair<KeyPoint *, KeyPoint *>> &pairs, const string& windowName) {
    cv::Mat im1 = cv::imread(firstImagePath, 1);
    cv::Mat im2 = cv::imread(secondImagePath, 1);
    bool hstack = im1.cols < im1.rows;
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

vector<pair<KeyPoint *, KeyPoint *>> ImageAnalyzer::getNPairs(int n, vector<pair<KeyPoint*, KeyPoint*>>& distribution) {
    vector<pair<KeyPoint*, KeyPoint*>> randomKeyPoints;
    if(ransacHeuristic == Distance){
        randomKeyPoints = getNDifferentKeyPointPairsHeuristic(n, distribution);
    } else {
        randomKeyPoints = getNDifferentKeyPointPairs(n, distribution);
    }
    return randomKeyPoints;
}

vector<pair<KeyPoint*, KeyPoint*>> ImageAnalyzer::getNDifferentKeyPointPairs(int n, vector<pair<KeyPoint*, KeyPoint*>>& distribution) {
    unordered_set<int> indices;
    std::uniform_int_distribution<int> indexDistribution(0, distribution.size() -1);
    while(indices.size() != n) {
        int randomIndex = indexDistribution(randomEngine);
        indices.insert(randomIndex);
    }
    vector<pair<KeyPoint*, KeyPoint*>> result;
    for(int index : indices){
        result.push_back(distribution[index]);
    }
    return result;
}

vector<pair<KeyPoint *, KeyPoint *>> ImageAnalyzer::getNDifferentKeyPointPairsHeuristic(int n, vector<pair<KeyPoint*, KeyPoint*>>& distribution) {
    vector<pair<KeyPoint*, KeyPoint*>> result;
    vector<pair<KeyPoint*, KeyPoint*>> filteredDistribution = distribution;
    while(result.size() != n) {
        std::uniform_int_distribution<int> indexDistribution(0, filteredDistribution.size() -1);
        int randomIndex = indexDistribution(randomEngine);
        pair<KeyPoint*, KeyPoint*> randomPair = filteredDistribution[randomIndex];
        filteredDistribution = this->filterIncorrectPairs(randomPair, filteredDistribution);
        if (!filteredDistribution.empty()){
            result.push_back(randomPair);
        } else {
            filteredDistribution = distribution;
            result.clear();
        }
    }
    return result;
}

void ImageAnalyzer::runRansac() {
    if(keyPointPairs.size() < 3){
        matchingTransformKeyPointPairs = keyPointPairs;
    } else {
        if(ransacHeuristic == Iterations){
            estimateRansacIterations();
            std::cout << "Estimated ransac iterations: " << this->ransacIterations << std::endl;
        }
        runRansacImpl();
    }
}

void ImageAnalyzer::runRansacImpl() {
    Eigen::Matrix3d bestTransformation;
    vector<pair<KeyPoint*, KeyPoint*>> bestConsensus;
    vector<pair<KeyPoint*, KeyPoint*>> pairDistribution(this->keyPointPairs);
    int bestScore = 0;
    for(int i = 0; i < ransacIterations; i++){
        Eigen::MatrixXd A;
        vector<pair<KeyPoint*, KeyPoint*>> pairSample;
        if(transformationType == Affine){
            pairSample = getNPairs(3, pairDistribution);
            A = getAffineTransform(pairSample);
        } else {
            pairSample = getNPairs(4, pairDistribution);
            A = getPerspectiveTransform(pairSample);
        }
        vector<pair<KeyPoint*, KeyPoint*>> consensus;
        consensus.reserve(keyPointPairs.size());
        vector<int> distributionIndices;
        for(auto& pair : keyPointPairs){
            Eigen::Vector3d point;
            point << pair.first->getX(),
                    pair.first->getY(),
                    1;
            Eigen::Vector3d transformedPoint = A * point;
            transformedPoint(0) = transformedPoint(0) / transformedPoint(2);
            transformedPoint(1) = transformedPoint(1) / transformedPoint(2);
            double distance = pair.second->euclideanDistance(transformedPoint(0), transformedPoint(1));
            if(distance < transformationErrorThreshold){
                consensus.push_back(pair);
            }
        }
        if(consensus.size() > bestScore){
            bestScore = consensus.size();
            bestTransformation = A;
            bestConsensus = consensus;
            if (this->ransacHeuristic == Distribution) {
                pairDistribution.insert(pairDistribution.end(), pairSample.begin(), pairSample.end());
            }
            std::cout << "iter: " << i << " current score: " << bestScore << "best possible: " << keyPointPairs.size() << std::endl;
        }
    }
    this->bestFoundTransformation = bestTransformation;
    this->matchingTransformKeyPointPairs = bestConsensus;
}

Eigen::MatrixXd ImageAnalyzer::getAffineTransform(vector<pair<KeyPoint*, KeyPoint*>>& pairSample) {
    KeyPoint *firstPointA = pairSample[0].first;
    KeyPoint *secondPointA = pairSample[1].first;
    KeyPoint *thirdPointA = pairSample[2].first;
    KeyPoint *firstPointB = pairSample[0].second;
    KeyPoint *secondPointB = pairSample[1].second;
    KeyPoint *thirdPointB = pairSample[2].second;
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

Eigen::MatrixXd ImageAnalyzer::getPerspectiveTransform(vector<pair<KeyPoint*, KeyPoint*>>& pairSample) {
    KeyPoint *A1 = pairSample[0].first;
    KeyPoint *A2 = pairSample[1].first;
    KeyPoint *A3 = pairSample[2].first;
    KeyPoint *A4 = pairSample[3].first;
    KeyPoint *B1 = pairSample[0].second;
    KeyPoint *B2 = pairSample[1].second;
    KeyPoint *B3 = pairSample[2].second;
    KeyPoint *B4 = pairSample[3].second;
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
            B3->getY(),
            B4->getY();
    Eigen::VectorXd vec = X.inverse() * Y;
    Eigen::MatrixXd result(3,3);
    result << vec(0), vec(1), vec(2),
            vec(3), vec(4), vec(5),
            vec(6), vec(7), 1;
    return result;
}

bool ImageAnalyzer::distanceHeuristicCorrect(pair<KeyPoint *, KeyPoint *> &firstPair,
                                             pair<KeyPoint *, KeyPoint *> &secondPair) {
    double firstPointsLen2 = firstPair.first->squaredEuclideanDistance(*secondPair.first);
    if(smallRSquared < firstPointsLen2 && firstPointsLen2 < bigRSquared){
        double secondPointsLen2 = firstPair.second->squaredEuclideanDistance(*secondPair.second);
        return smallRSquared < secondPointsLen2 && secondPointsLen2 < bigRSquared;
    } else {
        return false;
    }
}

void ImageAnalyzer::initDistanceHeuristic() {
    cv::Mat first = cv::imread(firstImagePath);
    int size = std::max(first.rows, first.cols);
    smallRSquared = pow(size * 0.01f, 2);
    bigRSquared = pow(size * 0.3f, 2);
    std::cout << smallRSquared << " " << bigRSquared << std::endl;
}

void ImageAnalyzer::estimateRansacIterations() {
    int n;
    if(this->transformationType == Affine){
        n = 3;
    } else {
        n = 4;
    }
    double w;
    if(keyPointPairs.size() == 0){
        this->ransacIterations = 0;
    } else {
        w = coherentKeyPointPairs.size() / (double)keyPointPairs.size();
        double denom = log2(1 - pow(w, n));
        if(denom == 0) {
            this->ransacIterations = 0;
        }
        double result = log2(1 - ransacProbability) / denom;
        this->ransacIterations = (int)result;
    }
}

vector<pair<KeyPoint *, KeyPoint *>>
ImageAnalyzer::filterIncorrectPairs(pair<KeyPoint *, KeyPoint *> newPair, vector<pair<KeyPoint *, KeyPoint *>> allPairs) {
    vector<pair<KeyPoint *, KeyPoint *>> result;
    std::copy_if(allPairs.begin(), allPairs.end(), std::back_inserter(result), [&](pair<KeyPoint*, KeyPoint*> pair){ return distanceHeuristicCorrect(newPair, pair); });
    return result;
}
