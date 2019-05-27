//
// Created by Szymon on 26.05.2019.
//
#include "vector"
#include "string"
#include "utility"
#include "fstream"
#include "sstream"
#include <iterator>
#include <algorithm>
#include <limits>
#include <cmath>

using std::vector;
using std::string;
using std::pair;
using std::abs;
#ifndef INC_4_IMAGE_SIMILARITY_C_KEYPOINT_H
#define INC_4_IMAGE_SIMILARITY_C_KEYPOINT_H


class KeyPoint {
public:
    KeyPoint(double x, double y, vector<double> features);
    double featureDistance(KeyPoint& other);
    double squaredEuclideanDistance(double otherX, double otherY);
    double euclideanDistance(KeyPoint& other);
    double euclideanDistance(double otherX, double otherY);
    double squaredEuclideanDistance(KeyPoint &other);
    KeyPoint* getClosest(vector<KeyPoint*>& keyPoints);
    void calculateNeighbourhood(vector<KeyPoint*>& allImageKeyPoints, int neighbourhoodSize);
    bool neighbourhoodContains(KeyPoint* otherKeyPoint);

    double getX() const;

    double getY() const;

    static KeyPoint* fromString(string& text);
    static vector<KeyPoint*> importKeyPoints(const string& fileName);
    static vector<pair<KeyPoint*, KeyPoint*>> getKeyPointPairs(vector<KeyPoint*>& firstKeyPoints, vector<KeyPoint*>& secondKeyPoints);
private:
    double x;
    double y;
    vector<double> features;
    vector<KeyPoint*> neighbourhood;
    double len2;

    static vector<string> splitInputLine(string &line);
};


#endif //INC_4_IMAGE_SIMILARITY_C_KEYPOINT_H
