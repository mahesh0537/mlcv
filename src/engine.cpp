#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
// #include <opencv2/dnn/dnn.hpp>
#include "engine.h"
#include "NvOnnxParser.h"

void Logger::log(Severity severity, const char *msg) noexcept {
    // Would advise using a proper logging utility such as https://github.com/gabime/spdlog
    // For the sake of this tutorial, will just log to the console.

    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

bool Engine::doesFileExist(const std::string &filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

Engine::Engine(const Options &options)
    : m_options(options) {}

bool Engine::build(std::string onnxModelPath) {
    // Only regenerate the engine file if it has not already been generated for the specified options
    m_engineName = serializeEngineOptions(m_options);
    std::cout << "Searching for engine file with name: " << m_engineName << std::endl;

    if (doesFileExist(m_engineName)) {
        std::cout << "Engine found, not regenerating..." << std::endl;
        return true;
    }

    // Was not able to find the engine file, generate...
    std::cout << "Engine not found, generating..." << std::endl;

    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Set the max supported batch size
    builder->setMaxBatchSize(m_options.maxBatchSize);

    // Define an explicit batch size and then create the network.
    // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    // We are going to first read the onnx file into memory, then pass that buffer to the parser.
    // Had our onnx model file been encrypted, this approach would allow us to first decrypt the buffer.

    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    // Save the input height, width, and channels.
    // Require this info for inference.
    const auto input = network->getInput(0);
    const auto output = network->getOutput(0);
    const auto inputName = input->getName();
    const auto inputDims = input->getDimensions();
  int32_t inputC = inputDims.d[1];
  int32_t inputH = inputDims.d[2];
  int32_t inputW = inputDims.d[3];

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // Specify the optimization profiles and the
    IOptimizationProfile* defaultProfile = builder->createOptimizationProfile();
    defaultProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
    defaultProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(1, inputC, inputH, inputW));
    defaultProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
    config->addOptimizationProfile(defaultProfile);

    // Specify all the optimization profiles.
    for (const auto& optBatchSize: m_options.optBatchSizes) {
        if (optBatchSize == 1) {
            continue;
        }

        if (optBatchSize > m_options.maxBatchSize) {
            throw std::runtime_error("optBatchSize cannot be greater than maxBatchSize!");
        }

        IOptimizationProfile* profile = builder->createOptimizationProfile();
        profile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
        profile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(optBatchSize, inputC, inputH, inputW));
        profile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
        config->addOptimizationProfile(profile);
    }

    config->setMaxWorkspaceSize(m_options.maxWorkspaceSize);

    if (m_options.FP16) {
        config->setFlag(BuilderFlag::kFP16);
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream) {
        return false;
    }
    config->setProfileStream(*profileStream);

    // Build the engine
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    // Write the engine to disk
    std::ofstream outfile(m_engineName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    std::cout << "Success, saved engine to " << m_engineName << std::endl;

    return true;
}

Engine::~Engine() {
    if (m_cudaStream) {
        cudaStreamDestroy(m_cudaStream);
    }
}

bool Engine::loadNetwork() {
    // Read the serialized model from disk
    std::ifstream file(m_engineName, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    std::unique_ptr<IRuntime> runtime{createInferRuntime(m_logger)};
    if (!runtime) {
        return false;
    }

    // Set the device index
    auto ret = cudaSetDevice(m_options.deviceIndex);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) +
                ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    auto cudaRet = cudaStreamCreate(&m_cudaStream);
    if (cudaRet != 0) {
        throw std::runtime_error("Unable to create cuda stream");
    }

    return true;
}

bool Engine::runInference(const std::vector<cv::Mat> &inputFaceChips, std::vector<cv::Rect>& featureVectors, int img_w, int img_h) {
    float conf_th = 0.1;
    auto dims = m_engine->getBindingDimensions(0);
    auto output_bbox = m_engine->getBindingDimensions(1);
    auto output_score = m_engine->getBindingDimensions(2);
    // std::cout<<output_bbox<<std::endl;
    // std::cout<<output_score<<std::endl;
    // Dims4 inputDims = {static_cast<int32_t>(inputFaceChips.size()), dims.d[1], dims.d[2], dims.d[3]};
    Dims4 inputDims = {dims.d[0], dims.d[1], dims.d[2], dims.d[3]};
    int iw = dims.d[3];
    int ih = dims.d[2];
    // std::cout<<inputDims<<std::endl;
        // std::abort();

    // m_context->setBindingDimensions(0, inputDims);

    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all input dimensions specified.");
    }
    // // else{
    // //     std::cout<<"all done till dimension"<<std::endl;
    // // }

    auto batchSize = static_cast<int32_t>(inputFaceChips.size());
    // // Only reallocate buffers if the batch size has changed
    if (m_prevBatchSize != inputFaceChips.size()) {
        std::cout<<"change in batch size"<<std::endl;
        m_inputBuff.hostBuffer.resize(inputDims);
        m_inputBuff.deviceBuffer.resize(inputDims);

        Dims4 outputDims_bbox {output_bbox.d[0], output_bbox.d[1], output_bbox.d[2], output_bbox.d[3]};
        Dims3 outputDims_score {output_score.d[0], output_score.d[1], output_score.d[2]};

        m_outputBuff_bbox.hostBuffer.resize(outputDims_bbox);
        m_outputBuff_bbox.deviceBuffer.resize(outputDims_bbox);

        m_outputBuff_score.hostBuffer.resize(outputDims_score);
        m_outputBuff_score.deviceBuffer.resize(outputDims_score);

        m_prevBatchSize = batchSize;
    }
    int num_classes = output_score.d[2];
    int rows = output_score.d[1];
    // std::cout<<num_classes<<"num of class;"<<rows<<std::endl;

    auto* hostDataBuffer = static_cast<float*>(m_inputBuff.hostBuffer.data());

    for (size_t batch = 0; batch < inputFaceChips.size(); ++batch) {
        auto image = inputFaceChips[batch];

        // Preprocess code
        image.convertTo(image, CV_32FC3, 1.f / 255.f);
        // cv::subtract(image, cv::Scalar(0.5f, 0.5f, 0.5f), image, cv::noArray(), -1);
        // cv::divide(image, cv::Scalar(0.5f, 0.5f, 0.5f), image, 1, -1);

        // NHWC to NCHW conversion
        // NHWC: For each pixel, its 3 colors are stored together in RGB order.
        // For a 3 channel image, say RGB, pixels of the R channel are stored first, then the G channel and finally the B channel.
        // https://user-images.githubusercontent.com/20233731/85104458-3928a100-b23b-11ea-9e7e-95da726fef92.png
        int offset = dims.d[1] * dims.d[2] * dims.d[3] * batch;
        // offset = 0;
        int r = 0 , g = 0, b = 0;
        // std::cout<<dims.d[2] * dims.d[3]<<std::endl;
        for (int i = 0; i < dims.d[1] * dims.d[2] * dims.d[3]; ++i) {
            // hostDataBuffer[offset + i] = *(reinterpret_cast<float*>(image.data) + i);
            if (i % 3 == 0) {
                hostDataBuffer[offset + r++] = *(reinterpret_cast<float*>(image.data) + i);
                // std::cout<<*(reinterpret_cast<float*>(image.data) + i)<<std::endl;
            } else if (i % 3 == 1) {
                hostDataBuffer[offset + g++ + dims.d[2]*dims.d[3]] = *(reinterpret_cast<float*>(image.data) + i);
            } else {
                hostDataBuffer[offset + b++ + dims.d[2]*dims.d[3]*2] = *(reinterpret_cast<float*>(image.data) + i);
            }
            // std::cout<<hostDataBuffer[offset + r]<<std::endl;
            // std::cout<<i<<std::endl;
            // std::cout<<r<<"r"<<std::endl;
            // std::cout<<g<<"g"<<std::endl;
            // std::cout<<b<<"b"<<std::endl;

        }
        // std::cout<<r<<std::endl;
    }

    // // Copy from CPU to GPU
    auto ret = cudaMemcpyAsync(m_inputBuff.deviceBuffer.data(), m_inputBuff.hostBuffer.data(), m_inputBuff.hostBuffer.nbBytes(), cudaMemcpyHostToDevice, m_cudaStream);
    // std::cout<<ret;
    if (ret != 0) {
        std::cout<<"unable to copy from CPU to GPU"<<std::endl;
        return false;
    }

    std::vector<void*> predicitonBindings = {m_inputBuff.deviceBuffer.data(), m_outputBuff_bbox.deviceBuffer.data(), m_outputBuff_score.deviceBuffer.data()};
    //     // std::vector<void*> predicitonBindings = {m_inputBuff.deviceBuffer.data()};


    // // // Run inference.
    bool status = m_context->enqueueV2(predicitonBindings.data(), m_cudaStream, nullptr);

    if (!status) {
        std::cout<<"unable to do inference"<<std::endl;
        return false;
    }
    // std::cout<<"inference done"<<std::endl;
    // // // Copy the results back to CPU memory
    ret = cudaMemcpyAsync(m_outputBuff_bbox.hostBuffer.data(), m_outputBuff_bbox.deviceBuffer.data(), m_outputBuff_bbox.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost, m_cudaStream);
    if (ret != 0) {
        std::cout << "Unable to copy buffer from GPU back to CPU" << std::endl;
        return false;
    }

    ret = cudaMemcpyAsync(m_outputBuff_score.hostBuffer.data(), m_outputBuff_score.deviceBuffer.data(), m_outputBuff_score.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost, m_cudaStream);
    if (ret != 0) {
        std::cout << "Unable to copy buffer from GPU back to CPU" << std::endl;
        return false;
    }

    ret = cudaStreamSynchronize(m_cudaStream);
    if (ret != 0) {
        std::cout << "Unable to synchronize cuda stream" << std::endl;
        return false;
    }


    const float * score = reinterpret_cast<const float*>(m_outputBuff_score.hostBuffer.data());
    const float * bbox = reinterpret_cast<const float*>(m_outputBuff_bbox.hostBuffer.data());

    // std::cout<<score;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    float scalex = 1.0 / (img_w * 1.0);
    float scaley =  1.0 / (img_h * 1.0);


    for(int i = 0; i < rows; ++i){
                    // std::cout<<i;

        if(score[i] > conf_th){
        // std::cout<<score[i]<<std::endl;

            // const float * classes_scores = score + i;
            // cv::Mat scores(1, 1, CV_32FC1, classes_scores);
            // cv::Point class_id;
            // double max_class_score;
            // minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // std::cout<<scale;
            float x0 = (bbox[i * 4]) / scalex;
            float y0 = (bbox[i * 4 + 1]) / scaley;
            float x1 = (bbox[i * 4 + 2]) / scalex;
            float y1 = (bbox[i * 4 + 3]) / scaley;
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
            // std::cout<<x0<<" "<< y0<<" "<<x1<<" "<<y1<<std::endl;
            // std::cout<<score[i]<<std::endl;
            boxes.push_back(cv::Rect(x0, y0, x1 - x0, y1 - y0));
            confidences.push_back(score[i]);

        }
    }
    // std::cout<<boxes[0]<<std::endl;

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.1, 0.1, indices);
    for (size_t i = 0; i < indices.size(); ++i){
        int idx = indices[i];
        cv::Rect r = boxes[idx];
        // std::cout<<r.x<<" "<< r.y<<" "<<r.width<<" "<<r.height<<std::endl;
        featureVectors.push_back(r);
    }
    // std::cout<<score.rows;

    // std::cout<<reinterpret_cast<const float*>(m_outputBuff_score.hostBuffer.data())[0]<<std::endl;
    // // // Copy to output
    // for (int batch = 0; batch < batchSize; ++batch) {
    //     std::vector<float> featureVector;
    //     featureVector.resize(outputL);

    //     memcpy(featureVector.data(), reinterpret_cast<const char*>(m_outputBuff.hostBuffer.data()) +
    //     batch * outputL * sizeof(float), outputL * sizeof(float ));
    //     featureVectors.emplace_back(std::move(featureVector));
    // }

    return true;
}

std::string Engine::serializeEngineOptions(const Options &options) {
    std::string engineName = "trt.engine";

    std::vector<std::string> gpuUUIDs;
    getGPUUUIDs(gpuUUIDs);

    if (static_cast<size_t>(options.deviceIndex) >= gpuUUIDs.size()) {
        throw std::runtime_error("Error, provided device index is out of range!");
    }

    engineName+= "." + gpuUUIDs[options.deviceIndex];

    // Serialize the specified options into the filename
    if (options.FP16) {
        engineName += ".fp16";
    } else {
        engineName += ".fp32";
    }

    engineName += "." + std::to_string(options.maxBatchSize) + ".";
    for (size_t i = 0; i < m_options.optBatchSizes.size(); ++i) {
        engineName += std::to_string(m_options.optBatchSizes[i]);
        if (i != m_options.optBatchSizes.size() - 1) {
            engineName += "_";
        }
    }

    engineName += "." + std::to_string(options.maxWorkspaceSize);

    return engineName;
}

void Engine::getGPUUUIDs(std::vector<std::string>& gpuUUIDs) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    for (int device=0; device<numGPUs; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        char uuid[33];
        for(int b=0; b<16; b++) {
            sprintf(&uuid[b*2], "%02x", (unsigned char)prop.uuid.bytes[b]);
        }

        gpuUUIDs.push_back(std::string(uuid));
        // by comparing uuid against a preset list of valid uuids given by the client (using: nvidia-smi -L) we decide which gpus can be used.
    }
}