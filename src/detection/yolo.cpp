#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

const float CONFIDENCE_THRESHOLD = 0.25f;
const float NMS_THRESHOLD = 0.45f;
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;

//------------------------------------------------------------
// Function Declarations
//------------------------------------------------------------
std::vector<std::string> loadClasses(const std::string& fileName);
cv::dnn::Net loadModel(const std::string& modelPath);
std::string getAutomatedVideoName();
std::vector<fs::path> getFrames(const fs::path& folder);
cv::Mat createBlob(const cv::Mat& image);

void decodeAndDrawYOLOOutput(
    cv::Mat& image,
    const cv::Mat& output,
    const std::vector<std::string>& classNames,
    json& frameObjectsArray
);

void processFrame(
    const fs::path& imagePath,
    cv::dnn::Net& net,
    const std::vector<std::string>& classNames,
    const std::string& outputFolder,
    json& frameObjectsArray
);

//------------------------------------------------------------
// Load COCO Class Names
//------------------------------------------------------------
std::vector<std::string> loadClasses(const std::string& fileName)
{
    std::vector<std::string> classes;
    std::ifstream file(fileName);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open coco.names");
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty())
            classes.push_back(line);
    }

    return classes;
}

//------------------------------------------------------------
// Load YOLO ONNX Model
//------------------------------------------------------------
cv::dnn::Net loadModel(const std::string& modelPath)
{
    cv::dnn::Net net = cv::dnn::readNet(modelPath);

    if (net.empty())
    {
        throw std::runtime_error("Could not load YOLO11 ONNX model.");
    }

    return net;
}

//------------------------------------------------------------
// Fetch Session Configuration Video Token
//------------------------------------------------------------
std::string getAutomatedVideoName()
{
    std::ifstream file("Configs/pipeline_config.txt");
    if (!file.is_open())
    {
        throw std::runtime_error("Pipeline configuration missing! Run video_ingestion first.");
    }
    std::string videoName;
    std::getline(file, videoName);
    return videoName;
}

//------------------------------------------------------------
// Get Frames from Target Subdirectory
//------------------------------------------------------------
std::vector<fs::path> getFrames(const fs::path& folder)
{
    std::vector<fs::path> frames;

    if (!fs::exists(folder))
    {
        throw std::runtime_error("Target frame generation folder missing: " + folder.string());
    }

    for (const auto& entry : fs::directory_iterator(folder))
    {
        if (entry.path().extension() == ".jpg")
            frames.push_back(entry.path());
    }

    std::sort(frames.begin(), frames.end());

    return frames;
}

//------------------------------------------------------------
// Create Blob from Image Matrix
//------------------------------------------------------------
cv::Mat createBlob(const cv::Mat& image)
{
    cv::Mat blob;

    cv::dnn::blobFromImage(
        image,
        blob,
        1.0 / 255.0,
        cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
        cv::Scalar(),
        true,
        false
    );

    return blob;
}

//------------------------------------------------------------
// Decode Output, Draw Boxes, and Capture Data for JSON Objects
//------------------------------------------------------------
void decodeAndDrawYOLOOutput(
    cv::Mat& image,
    const cv::Mat& output,
    const std::vector<std::string>& classNames,
    json& frameObjectsArray
)
{
    // Reshape output layout configurations
    cv::Mat outputMat(
        output.size[1],
        output.size[2],
        CV_32F,
        (void*)output.ptr<float>()
    );

    cv::transpose(outputMat, outputMat);

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    cv::Size imageSize = image.size();

    for (int i = 0; i < outputMat.rows; i++)
    {
        float* data = outputMat.ptr<float>(i);

        float cx = data[0];
        float cy = data[1];
        float w  = data[2];
        float h  = data[3];

        float maxScore = 0.0f;
        int classId = -1;

        for (int c = 4; c < 84; c++)
        {
            if (data[c] > maxScore)
            {
                maxScore = data[c];
                classId = c - 4;
            }
        }

        if (maxScore < CONFIDENCE_THRESHOLD)
            continue;

        int left   = static_cast<int>((cx - w / 2) * imageSize.width / INPUT_WIDTH);
        int top    = static_cast<int>((cy - h / 2) * imageSize.height / INPUT_HEIGHT);
        int width  = static_cast<int>(w * imageSize.width / INPUT_WIDTH);
        int height = static_cast<int>(h * imageSize.height / INPUT_HEIGHT);

        boxes.emplace_back(left, top, width, height);
        confidences.push_back(maxScore);
        classIds.push_back(classId);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(
        boxes,
        confidences,
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD,
        indices
    );

    std::cout << "Detections After NMS : " << indices.size() << std::endl;

    // Process each valid detection item sequence
    for (int index : indices)
    {
        cv::rectangle(image, boxes[index], cv::Scalar(0, 255, 0), 2);

        std::string label =
            classNames[classIds[index]] +
            " " +
            cv::format("%.2f", confidences[index]);

        cv::putText(
            image,
            label,
            cv::Point(boxes[index].x, boxes[index].y - 8),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 255, 0),
            2
        );

        // Build object layout nodes for JSON output metrics mapping
        json detectionItem;
        detectionItem["class"] = classNames[classIds[index]];
        detectionItem["confidence"] = std::round(confidences[index] * 100.0) / 100.0;
        detectionItem["bbox"] = {
            {"x", boxes[index].x},
            {"y", boxes[index].y},
            {"width", boxes[index].width},
            {"height", boxes[index].height}
        };
        frameObjectsArray.push_back(detectionItem);
    }
}

//------------------------------------------------------------
// Process Single Selected Frame Pipeline
//------------------------------------------------------------
void processFrame(
    const fs::path& imagePath,
    cv::dnn::Net& net,
    const std::vector<std::string>& classNames,
    const std::string& outputFolder,
    json& frameObjectsArray
)
{
    cv::Mat image = cv::imread(imagePath.string());
    if (image.empty())
    {
        std::cout << "WARNING: Frame unreadable: " << imagePath.filename().string() << std::endl;
        return;
    }

    cv::Mat blob = createBlob(image);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    decodeAndDrawYOLOOutput(image, outputs[0], classNames, frameObjectsArray);

    std::string outputFile = outputFolder + "/" + imagePath.filename().string();
    cv::imwrite(outputFile, image);

    std::cout << "Saved Annotated Image : " << outputFile << std::endl;
}

//------------------------------------------------------------
// Main Workflow Execution
//------------------------------------------------------------
int main()
{
    try
    {
        // 1. Load Setup Configurations
        auto classNames = loadClasses("Configs/coco.names");
        std::cout << "Loaded " << classNames.size() << " classes." << std::endl;

        auto net = loadModel("Models/yolo11s.onnx");
        std::cout << "YOLO11 Loaded Successfully." << std::endl;

        // 2. Automation Hook: Track target input folders directly from "Frames" directory
        std::string videoName = getAutomatedVideoName();
        fs::path inputFramesDir = "Frames";
        auto frames = getFrames(inputFramesDir);

        if (frames.empty())
        {
            throw std::runtime_error("No extraction images found inside directory: " + inputFramesDir.string());
        }

        std::cout << "\nFrames Found for YOLO Processing: " << frames.size() << std::endl;

        // 3. Setup Annotated Destination Folder directly under "annotated_frames" root directory
        std::string outputFolder = "annotated_frames";
        if (!fs::exists(outputFolder))
        {
            fs::create_directories(outputFolder);
        }
        std::cout << "Output Annotated Directory Target: " << outputFolder << std::endl;

        // 4. Setup JSON Telemetry Directory
        std::string detectionsDir = "detections";
        if (!fs::exists(detectionsDir))
        {
            fs::create_directories(detectionsDir);
        }
        std::string jsonPathName = detectionsDir + "/detections.json";

        // 5. Construct Explicit Target Step Indexing List (0, 4, 9, 14, 19... for frames 1, 5, 10...)
        std::vector<size_t> frameIndices;
        if (!frames.empty())
        {
            frameIndices.push_back(0); 
        }
        for (size_t idx = 4; idx < frames.size(); idx += 5)
        {
            frameIndices.push_back(idx); 
        }

        json detectionFramesArray = json::array();
        std::cout << "Beginning AI pipeline processing loops...\n" << std::endl;

        // 6. Execution Loop Core 
        for (size_t idx : frameIndices)
        {
            fs::path currentFramePath = frames[idx];
            std::cout << "\n[Processing Frame Index: " << idx << "] Target: " 
                      << currentFramePath.filename().string() << std::endl;

            json frameNode;
            frameNode["frame"] = currentFramePath.filename().string();
            frameNode["objects"] = json::array();

            processFrame(currentFramePath, net, classNames, outputFolder, frameNode["objects"]);

            detectionFramesArray.push_back(frameNode);
        }

        // 7. Assemble and Write Singular Root JSON Document Matrix
        json rootJson;
        rootJson["video_token"] = videoName;
        rootJson["detections_log"] = detectionFramesArray;

        std::ofstream file(jsonPathName);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not generate target detections compilation file: " + jsonPathName);
        }
        file << rootJson.dump(4);
        file.close();

        std::cout << "\n==========================================" << std::endl;
        std::cout << "SUCCESS: Automated Batch Execution Complete!" << std::endl;
        std::cout << "Saved Annotated Images to : " << outputFolder << std::endl;
        std::cout << "Detections Array Log Saved to: " << jsonPathName << std::endl;
        std::cout << "==========================================" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "\nYOLO MODULE FATAL ERROR : " << e.what() << std::endl;
        return -1;
    }

    return 0;
}