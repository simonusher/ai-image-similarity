#include <utility>

//
// Created by Szymon on 26.05.2019.
//

#include "KeyPoint.h"

KeyPoint::KeyPoint(double x, double y, vector<double> features) {
    this->x = x;
    this->y = y;
    this->features = std::move(features);
}

double KeyPoint::featureDistance(KeyPoint &other) {
    double distance = 0;
    for(int i = 0; i < this->features.size(); i++){
        distance += abs(this->features[i] - other.features[i]);
    }
    return distance;
}

KeyPoint* KeyPoint::getClosest(vector<KeyPoint *> &keyPoints) {
    double minDistance = std::numeric_limits<double>::max();
    KeyPoint* closest = nullptr;
    for(int i = 0; i < keyPoints.size(); i++){
        double distance = this->featureDistance(*keyPoints[i]);
        if(distance < minDistance){
            minDistance = distance;
            closest = keyPoints[i];
        }
    }
    return closest;
}

KeyPoint* KeyPoint::fromString(string &text) {
    vector<string> tokens = splitInputLine(text);
    double x = std::stod(tokens[0]);
    double y = std::stod(tokens[1]);
    int nFeatures = tokens.size() - 5;
    vector<double> features;
    features.reserve(nFeatures);
    for(int i = 5; i < tokens.size(); i++) {
        features.push_back(std::stod(tokens[i]));
    }
    return new KeyPoint(x, y, features);
}

std::vector<std::string> KeyPoint::splitInputLine(std::string &line) {
    std::istringstream stringStream(line);
    std::vector<std::string> lineTokens;
    std::copy(std::istream_iterator<std::string>(stringStream),
              std::istream_iterator<std::string>(),
              std::back_inserter(lineTokens));
    return lineTokens;
}

vector<KeyPoint *> KeyPoint::importKeyPoints(const string &fileName) {
    string line;
    std::ifstream file;
    vector<KeyPoint*> keyPoints;
    file.open(fileName, std::ios::in);
    std::getline(file, line);
    std::getline(file, line);
    while(std::getline(file, line)){
        if(!line.empty()){
            keyPoints.push_back(fromString(line));
        }
    }
    return keyPoints;
}

vector<pair<KeyPoint *, KeyPoint *>>
KeyPoint::getKeyPointPairs(vector<KeyPoint *> &firstKeyPoints, vector<KeyPoint *> &secondKeyPoints) {
    vector<pair<KeyPoint*, KeyPoint*>> pairs;
    for(int i = 0; i < firstKeyPoints.size(); i++){
        KeyPoint* closestToFirst = firstKeyPoints[i]->getClosest(secondKeyPoints);
        KeyPoint* closestToSecond = closestToFirst->getClosest(firstKeyPoints);
        if(closestToSecond == firstKeyPoints[i]){
            pairs.emplace_back(closestToSecond, closestToFirst);
        }
    }
    return pairs;
}

double KeyPoint::squaredEuclideanDistance(KeyPoint &other){
    double xdiff = this->x - other.x;
    double ydiff = this->y - other.y;
    return xdiff * xdiff + ydiff * ydiff;
}

double KeyPoint::euclideanDistance(KeyPoint& other) {
    return sqrt(squaredEuclideanDistance(other));
}

void KeyPoint::calculateNeighbourhood(vector<KeyPoint *> &allImageKeyPoints, int neighbourhoodSize) {
    this->neighbourhood.clear();
    for (KeyPoint* otherKeyPoint: allImageKeyPoints) {
        otherKeyPoint->len2 = this->squaredEuclideanDistance(*otherKeyPoint);
    }
    std::sort(allImageKeyPoints.begin(), allImageKeyPoints.end(), [](const KeyPoint* first, const KeyPoint* second) {
        return first->len2 < second->len2;
    });
    this->neighbourhood = vector<KeyPoint*>();
    this->neighbourhood.reserve(neighbourhoodSize);
    for(int i = 0; neighbourhood.size() < neighbourhoodSize && i < allImageKeyPoints.size(); i++){
        if(allImageKeyPoints[i]->len2 > 0){
            this->neighbourhood.push_back(allImageKeyPoints[i]);
        }
    }
}

bool KeyPoint::neighbourhoodContains(KeyPoint *otherKeyPoint) {
    return std::find(this->neighbourhood.begin(), this->neighbourhood.end(), otherKeyPoint) != this->neighbourhood.end();
}

double KeyPoint::getX() const {
    return x;
}

double KeyPoint::getY() const {
    return y;
}
