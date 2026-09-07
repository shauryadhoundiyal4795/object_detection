#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

//------------------------------------------------------------
// Helper Function to Read Automated Video Token
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
// Load JSON Telemetry File
//------------------------------------------------------------
json loadTrackingJson(const std::string& jsonPath)
{
    std::ifstream file(jsonPath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open telemetry file: " + jsonPath);
    }

    json data;
    file >> data;
    return data;
}

//------------------------------------------------------------
// Main Tracking Visualizer Pipeline
//------------------------------------------------------------
int main()
{
    try
    {
        // 1. Fetch active session token and set up paths
        std::string videoName = getAutomatedVideoName();
        std::cout << "Automated Session Active. Selected Video Token: " << videoName << std::endl;

        fs::path annotatedFolder = "annotated_frames";
        std::string jsonPath = "tracking_result/tracking.json";

        std::cout << "Locating Telemetry Matrix: " << jsonPath << std::endl;
        json trackingData = loadTrackingJson(jsonPath);
        std::cout << "Successfully parsed tracking telemetry array data." << std::endl;

        // 2. Setup Flat Target Output Directory "tracked_frames" directly at root
        std::string outputDir = "tracked_frames";
        if (!fs::exists(outputDir))
        {
            fs::create_directories(outputDir);
        }
        std::cout << "Target Destination Path: " << outputDir << "\n" << std::endl;

        // 3. Iterate over the frame entries inside the root array of tracking.json
        for (const auto& frameEntry : trackingData)
        {
            std::string frameFileName = frameEntry.value("frame", "");
            if (frameFileName.empty()) continue;

            fs::path inputImagePath = annotatedFolder / frameFileName;
            std::cout << "Visualizing Persistent ID -> " << frameFileName << "..." << std::endl;

            // Read the already annotated image frame layout
            cv::Mat image = cv::imread(inputImagePath.string());
            if (image.empty())
            {
                std::cout << "WARNING: Frame unreadable or missing from flat folder: " << inputImagePath.string() << std::endl;
                continue;
            }

            // 4. Loop through and overlay the parsed track ID metrics
            if (frameEntry.contains("tracks") && frameEntry["tracks"].is_array())
            {
                for (const auto& track : frameEntry["tracks"])
                {
                    int trackId = track.value("track_id", -1);
                    
                    // Extract precise boundary variables from tracking.json
                    double x1 = track.value("x1", 0.0);
                    double y1 = track.value("y1", 0.0);
                    
                    int ix1 = static_cast<int>(x1);
                    int iy1 = static_cast<int>(y1);

                    // Build string parameters for the rendering step
                    std::string trackLabel = "ID: " + std::to_string(trackId);

                    // Configure visualization graphics theme (Bright Cyan text matching track IDs)
                    cv::Scalar idTextColor(255, 255, 0); 
                    cv::Scalar textBgColor(0, 0, 0);

                    // Measure size parameters for rendering background headers above the original text
                    int baseLine;
                    cv::Size labelSize = cv::getTextSize(trackLabel, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                    
                    // Adjust placement parameters to stack nicely above your existing YOLO labels
                    int placementY = std::max(iy1 - 28, labelSize.height + 4);

                    // Render solid color bounding backing block for ID metric text
                    cv::rectangle(
                        image,
                        cv::Point(ix1, placementY - labelSize.height - 2),
                        cv::Point(ix1 + labelSize.width + 6, placementY + baseLine),
                        textBgColor,
                        cv::FILLED
                    );

                    // Put the new Track ID overlay text sequence directly onto the frame
                    cv::putText(
                        image,
                        trackLabel,
                        cv::Point(ix1 + 3, placementY),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        idTextColor,
                        1,
                        cv::LINE_AA
                    );
                }
            }

            // 5. Save the frame directly under the flat 'tracked_frames' directory root folder
            std::string outputFilePath = outputDir + "/" + frameFileName;
            cv::imwrite(outputFilePath, image);
            std::cout << "Saved Visualized Core Frame: " << outputFilePath << std::endl;
        }

        std::cout << "\n==========================================" << std::endl;
        std::cout << "VISUALIZATION SUCCESS: ID Pipeline Execution Complete!" << std::endl;
        std::cout << "All files verified directly inside: " << outputDir << std::endl;
        std::cout << "==========================================" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "\nVISUALIZER FATAL ERROR: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}