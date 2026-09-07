#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

int main()
{
    std::string videoFolder = "Videos";

    // Check whether the Videos folder exists
    if (!fs::exists(videoFolder))
    {
        std::cout << "Videos folder not found." << std::endl;
        return 1;
    }

    std::cout << "Videos folder found.\n" << std::endl;

    // Vector to store all valid video files
    std::vector<fs::path> videos;

    // Read all supported video files
    for (const auto& entry : fs::directory_iterator(videoFolder))
    {
        if (!entry.is_regular_file())
            continue;

        std::string extension = entry.path().extension().string();

        if (extension == ".mp4" ||
            extension == ".avi" ||
            extension == ".mov" ||
            extension == ".mkv")
        {
            videos.push_back(entry.path());
        }
    }

    // Check if any videos were found
    if (videos.empty())
    {
        std::cout << "No video files found." << std::endl;
        return 1;
    }

    // Sort videos alphabetically
    std::sort(videos.begin(), videos.end());

    // Display available videos
    std::cout << "Available Video Files:\n" << std::endl;

    for (size_t i = 0; i < videos.size(); i++)
    {
        std::cout << i + 1 << ". "
                  << videos[i].filename().string()
                  << std::endl;
    }

    // Ask the user to choose a video
    int choice;

    std::cout << "\nSelect a video by entering its number: ";
    std::cin >> choice;

    // Validate user input
    if (choice < 1 || choice > videos.size())
    {
        std::cout << "Invalid selection!" << std::endl;
        return 1;
    }

    // Store the selected video
    fs::path selectedVideo = videos[choice - 1];

    // Extract the video base name without extension (e.g., "test_02")
    std::string videoStem = selectedVideo.stem().string();

    std::cout << "\n====================================" << std::endl;
    std::cout << "Selected Video : " << selectedVideo.filename().string() << std::endl;
    std::cout << "Video Identifier: " << videoStem << std::endl;
    std::cout << "Full Path      : " << selectedVideo.string() << std::endl;
    std::cout << "====================================" << std::endl;

    // ------------------------------------------------------------------
    // AUTOMATION: Save choice to a central pipeline config file
    // ------------------------------------------------------------------
    try 
    {
        if (!fs::exists("Configs"))
        {
            fs::create_directories("Configs");
        }

        std::ofstream configFile("Configs/pipeline_config.txt");
        if (!configFile.is_open())
        {
            throw std::runtime_error("Unable to create pipeline_config.txt");
        }

        // Save the video name token down so the next stages can read it directly
        configFile << videoStem << std::endl;
        configFile.close();

        std::cout << "Pipeline session automated. Saved to Configs/pipeline_config.txt" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "Warning updating configuration token: " << e.what() << std::endl;
    }

    return 0;
}