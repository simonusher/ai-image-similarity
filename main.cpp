#include <iostream>
#include "KeyPoint.h"
#include "chrono"
#include "ImageAnalyzer.h"
#include <opencv2/opencv.hpp>
#include <cstdlib>
using namespace std;
int main() {
//    string firstFileName("img.png.haraff.sift");
//    string secondFileName("img2.png.haraff.sift");
//    vector<KeyPoint*> firstKeyPoints { KeyPoint::importKeyPoints(firstFileName) };
//    vector<KeyPoint*> secondKeyPoints { KeyPoint::importKeyPoints(secondFileName) };
//    ImageAnalyzer i(50, 0.6, firstKeyPoints, secondKeyPoints);
//    i.analyze();
//    std::cout << i.coherentKeyPointPairs.size();
    system("extract_features -haraff -sift -i morris.png -DE");
    cv::Mat im1 = cv::imread("morris.png", 1);
    cv::Mat im2 = cv::imread("morris.png", 1);
    cv::Mat imstack;
    cv::vconcat(im1, im2, imstack);
    cv::line(imstack, cv::Point(0, 0), cv::Point(50, 50), cv::Scalar(0, 0, 255));
    namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    imshow("Display Image", imstack);
    cv::waitKey(0);
    return 0;
}