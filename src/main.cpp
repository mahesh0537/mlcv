#include "engine.h"
#include <opencv2/opencv.hpp>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;


int main() {
    Options options;
    options.optBatchSizes = {1};

    Engine engine(options);

    // TODO: Specify your model here.
    // Must specify a dynamic batch size when exporting the model from onnx.
    const std::string onnxModelpath = "../yolov4_1_3_416_416_static.onnx";
    std::cout<<onnxModelpath<<std::endl;
    bool succ = engine.build(onnxModelpath);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }
    else{
        std::cout<<"TRT engine built"<<std::endl;
    }

    succ = engine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }
    if(succ){
        std::cout<<"Engine loaded"<<std::endl;
    }

    const size_t batchSize = 1;
    std::vector<cv::Mat> images;

    std::cout<<"Input in process"<<std::endl;
    const std::string inputImage = "../img.png";
    auto img = cv::imread(inputImage);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    int img_w = img.cols;
    int img_h = img.rows;
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(416,416));
    for (size_t i = 0; i < batchSize; i++) {
        images.push_back(img_resized);
    }
    

    std::cout<<"Input ready"<<std::endl;
    // Discard the first inference time as it takes longer
    std::vector<cv::Rect> featureVectors;
    succ = engine.runInference(images, featureVectors, img_w, img_h);
    if (!succ) {
        throw std::runtime_error("Unable to run inference.");
    }
    for (size_t i = 0; i < featureVectors.size(); ++i){
        cv::rectangle(img, featureVectors[i], cv::Scalar(0x27, 0xC1, 0x36), 2);
    }

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    cv::imwrite("result.jpg", img);

    size_t numIterations = 100000;

    auto t1 = Clock::now();
    for (size_t i = 0; i < numIterations; ++i) {
        featureVectors.clear();
        engine.runInference(images, featureVectors, img_w, img_h);
    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Success! Average time per inference: " << totalTime / numIterations / static_cast<float>(images.size()) <<
    " ms, for batch size of: " << images.size() << std::endl;

    return 0;
}
