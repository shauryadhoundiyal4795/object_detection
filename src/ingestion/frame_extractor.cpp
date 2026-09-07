#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

//------------------------------------------------------------
// Helper Function to Read Automated Video Token
//------------------------------------------------------------
std::string getAutomatedVideoName()
{
    std::ifstream file("Configs/pipeline_config.txt");
    if (!file.is_open())
    {
        throw std::runtime_error("Pipeline configuration missing! Please run video_ingestion first.");
    }
    std::string videoName;
    std::getline(file, videoName);
    return videoName;
}

//------------------------------------------------------------
// Main Execution Pipeline
//------------------------------------------------------------
int main()
{
    try
    {
        // 1. Automatically fetch the session's selected video name token
        std::string videoName = getAutomatedVideoName();
        std::cout << "Automated Session Active. Selected Video Token: " << videoName << std::endl;

        // 2. Scan the "Videos" folder natively to locate the exact file matching that name
        std::string videoFolder = "Videos";
        fs::path finalVideoPath;
        bool matchFound = false;

        for (const auto& entry : fs::directory_iterator(videoFolder))
        {
            if (entry.is_regular_file() && entry.path().stem().string() == videoName)
            {
                finalVideoPath = entry.path();
                matchFound = true;
                break;
            }
        }

        if (!matchFound)
        {
            throw std::runtime_error("Could not find a valid video file matching token: " + videoName);
        }

        std::cout << "Target File Located: " << finalVideoPath.string() << std::endl;

        // 3. Create Root Frames folder if it doesn't exist
        std::string framesRoot = "Frames";
        if (!fs::exists(framesRoot))
        {
            fs::create_directories(framesRoot);
            std::cout << "Created Frames root directory.\n";
        }

        // 4. Open video stream interface
        cv::VideoCapture video(finalVideoPath.string());
        if (!video.isOpened())
        {
            throw std::runtime_error("OpenCV Error: Could not open video file stream: " + finalVideoPath.string());
        }

        std::cout << "Video data streams opened successfully. Commencing extraction matrix...\n";

        cv::Mat frame;
        int frameNumber = 0;

        // 5. Run Frame Extraction Loop (Saving directly under 'Frames/')
        while (video.read(frame))
        {
            frameNumber++;
            std::stringstream filename;

            // Naming layout sequence saved directly to root folder: Frames/frame_000001.jpg
            filename << framesRoot
                     << "/frame_"
                     << std::setw(6)
                     << std::setfill('0')
                     << frameNumber
                     << ".jpg";

            cv::imwrite(filename.str(), frame);
            std::cout << "Saved Extracted Frame: " << filename.str() << std::endl;
        }

        video.release();

        std::cout << "\n==========================================" << std::endl;
        std::cout << "EXTRACTION COMPLETE: Automated Pipeline Linked!" << std::endl;
        std::cout << "Processed Source Video : " << finalVideoPath.filename().string() << std::endl;
        std::cout << "Total Frames Saved     : " << frameNumber << std::endl;
        std::cout << "Output Directory Location : " << framesRoot << std::endl;
        std::cout << "==========================================" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "\nFRAME EXTRACTOR FATAL ERROR: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}