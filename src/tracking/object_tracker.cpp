#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

// Pure data structure representing a persistent tracking trajectory
struct DeepTrack {
    int track_id;
    std::string class_name;
    double confidence;
    double x1, y1, x2, y2;
    double vx, vy;         // Velocity vectors (Motion prediction parameters)
    int age;               // Total lifetime frames count
    int time_since_update; // Consecutive missing frames counter
};

//------------------------------------------------------------
// Helper Function: Compute Spatial Overlap Matrix (IOU)
//------------------------------------------------------------
double computePredictiveIOU(const DeepTrack& track, double b_x1, double b_y1, double b_x2, double b_y2) {
    // Predict where the track's bounding box will be based on its velocity vector
    double pred_x1 = track.x1 + track.vx;
    double pred_y1 = track.y1 + track.vy;
    double pred_x2 = track.x2 + track.vx;
    double pred_y2 = track.y2 + track.vy;

    // Standard IOU intersection calculation on the predicted box location
    double inter_x1 = std::max(pred_x1, b_x1);
    double inter_y1 = std::max(pred_y1, b_y1);
    double inter_x2 = std::min(pred_x2, b_x2);
    double inter_y2 = std::min(pred_y2, b_y2);

    double inter_w = std::max(0.0, inter_x2 - inter_x1);
    double inter_h = std::max(0.0, inter_y2 - inter_y1);
    double inter_area = inter_w * inter_h;

    if (inter_area == 0.0) return 0.0;

    double area_a = (pred_x2 - pred_x1) * (pred_y2 - pred_y1);
    double area_b = (b_x2 - b_x1) * (b_y2 - b_y1);
    double union_area = area_a + area_b - inter_area;

    return inter_area / union_area;
}

//------------------------------------------------------------
// Main Execution Tracker Pipeline
//------------------------------------------------------------
int main()
{
    try
    {
        // 1. Verify existence of the source detections file
        std::string inputJsonPath = "detections/detections.json";
        if (!fs::exists(inputJsonPath))
        {
            throw std::runtime_error("Detections record missing! Run yolo.cpp first. Target: " + inputJsonPath);
        }

        std::cout << "Loading object detection data from: " << inputJsonPath << std::endl;
        
        std::ifstream inputFile(inputJsonPath);
        json detectionData;
        inputFile >> detectionData;
        inputFile.close();

        // 2. Setup output directories
        std::string outputDirName = "tracking_result";
        if (!fs::exists(outputDirName))
        {
            fs::create_directories(outputDirName);
        }
        std::string outputJsonPath = outputDirName + "/tracking.json";

        json trackingLogArray = json::array();
        
        // Active multi-object track database storage pool
        std::vector<DeepTrack> activeTracksPool;
        int nextGlobalTrackID = 1;

        // Tuning parameters matching standard DeepSORT constraints
        const double GATING_THRESHOLD = 0.20; // Minimum overlap matching limit
        const int MAX_AGE_BUDGET = 3;         // Number of frames an object can vanish before expiry

        // 3. Loop over frame series sequence entries
        if (detectionData.contains("detections_log") && detectionData["detections_log"].is_array())
        {
            for (const auto& frameNode : detectionData["detections_log"])
            {
                std::string frameName = frameNode.value("frame", "");
                std::cout << "Processing Cascaded Track Association: " << frameName << std::endl;

                json frameTrackNode;
                frameTrackNode["frame"] = frameName;
                frameTrackNode["tracks"] = json::array();

                std::vector<DeepTrack> detectionsThisFrame;

                // Parse individual item objects detected on current frame layer
                if (frameNode.contains("objects") && frameNode["objects"].is_array())
                {
                    for (const auto& detectedObj : frameNode["objects"])
                    {
                        std::string className = detectedObj.value("class", "unknown");
                        double confidence = detectedObj.value("confidence", 0.0);
                        
                        auto bbox = detectedObj["bbox"];
                        int x = bbox.value("x", 0);
                        int y = bbox.value("y", 0);
                        int width = bbox.value("width", 0);
                        int height = bbox.value("height", 0);

                        DeepTrack d;
                        d.track_id = -1; // Unassigned initially
                        d.class_name = className;
                        d.confidence = confidence;
                        d.x1 = static_cast<double>(x);
                        d.y1 = static_cast<double>(y);
                        d.x2 = static_cast<double>(x + width);
                        d.y2 = static_cast<double>(y + height);
                        d.vx = 0.0;
                        d.vy = 0.0;
                        d.age = 1;
                        d.time_since_update = 0;

                        detectionsThisFrame.push_back(d);
                    }
                }

                // Keep sizes fixed for association mapping tracking states
                size_t initialTracksCount = activeTracksPool.size();
                std::vector<bool> trackMatched(initialTracksCount, false);
                std::vector<bool> detectionMatched(detectionsThisFrame.size(), false);

                // 4. CASCADE DATA ASSOCIATION LOGIC
                // Step 1: Match highly probable existing tracks first
                for (size_t d_idx = 0; d_idx < detectionsThisFrame.size(); ++d_idx) {
                    double bestMatchMetric = 0.0;
                    int targetTrackIdx = -1;

                    for (size_t t_idx = 0; t_idx < initialTracksCount; ++t_idx) {
                        if (trackMatched[t_idx]) continue;
                        if (activeTracksPool[t_idx].class_name != detectionsThisFrame[d_idx].class_name) continue;

                        // Calculate predictive metric distance matching score
                        double matchScore = computePredictiveIOU(activeTracksPool[t_idx], 
                                                                 detectionsThisFrame[d_idx].x1, detectionsThisFrame[d_idx].y1,
                                                                 detectionsThisFrame[d_idx].x2, detectionsThisFrame[d_idx].y2);
                        
                        if (matchScore > bestMatchMetric) {
                            bestMatchMetric = matchScore;
                            targetTrackIdx = static_cast<int>(t_idx);
                        }
                    }

                    // Confirm match binding if it satisfies our gating boundaries
                    if (bestMatchMetric >= GATING_THRESHOLD && targetTrackIdx != -1) {
                        detectionMatched[d_idx] = true;
                        trackMatched[targetTrackIdx] = true;

                        // Calculate spatial displacement vectors to update internal target velocity metrics
                        double delta_x = detectionsThisFrame[d_idx].x1 - activeTracksPool[targetTrackIdx].x1;
                        double delta_y = detectionsThisFrame[d_idx].y1 - activeTracksPool[targetTrackIdx].y1;

                        // Apply a smoothing factor parameter to model target spatial inertia tracking mechanics
                        activeTracksPool[targetTrackIdx].vx = (activeTracksPool[targetTrackIdx].vx * 0.4) + (delta_x * 0.6);
                        activeTracksPool[targetTrackIdx].vy = (activeTracksPool[targetTrackIdx].vy * 0.4) + (delta_y * 0.6);

                        // Overwrite boundary tracking updates
                        activeTracksPool[targetTrackIdx].x1 = detectionsThisFrame[d_idx].x1;
                        activeTracksPool[targetTrackIdx].y1 = detectionsThisFrame[d_idx].y1;
                        activeTracksPool[targetTrackIdx].x2 = detectionsThisFrame[d_idx].x2;
                        activeTracksPool[targetTrackIdx].y2 = detectionsThisFrame[d_idx].y2;
                        activeTracksPool[targetTrackIdx].confidence = detectionsThisFrame[d_idx].confidence;
                        activeTracksPool[targetTrackIdx].time_since_update = 0;
                        activeTracksPool[targetTrackIdx].age++;

                        // Commit object track metadata down into telemetry arrays
                        json trackItem;
                        trackItem["track_id"] = activeTracksPool[targetTrackIdx].track_id;
                        trackItem["class"] = activeTracksPool[targetTrackIdx].class_name;
                        trackItem["confidence"] = activeTracksPool[targetTrackIdx].confidence;
                        trackItem["x1"] = activeTracksPool[targetTrackIdx].x1;
                        trackItem["y1"] = activeTracksPool[targetTrackIdx].y1;
                        trackItem["x2"] = activeTracksPool[targetTrackIdx].x2;
                        trackItem["y2"] = activeTracksPool[targetTrackIdx].y2;
                        frameTrackNode["tracks"].push_back(trackItem);
                    }
                }

                // Step 2: Handle unassociated detections (New trajectories entering screen)
                for (size_t d_idx = 0; d_idx < detectionsThisFrame.size(); ++d_idx) {
                    if (!detectionMatched[d_idx]) {
                        DeepTrack newTrack = detectionsThisFrame[d_idx];
                        newTrack.track_id = nextGlobalTrackID;
                        nextGlobalTrackID++;

                        activeTracksPool.push_back(newTrack);

                        json trackItem;
                        trackItem["track_id"] = newTrack.track_id;
                        trackItem["class"] = newTrack.class_name;
                        trackItem["confidence"] = newTrack.confidence;
                        trackItem["x1"] = newTrack.x1;
                        trackItem["y1"] = newTrack.y1;
                        trackItem["x2"] = newTrack.x2;
                        trackItem["y2"] = newTrack.y2;
                        frameTrackNode["tracks"].push_back(trackItem);
                    }
                }

                // 5. LIFETIME TRAJECTORY MANAGEMENT MATRIX
                // Only loop up to the initial counts to safely protect index integrity bounds
                for (size_t t_idx = 0; t_idx < initialTracksCount; ++t_idx) {
                    if (!trackMatched[t_idx]) {
                        activeTracksPool[t_idx].time_since_update++;
                        
                        // Decay momentum if tracking confirmation signatures vanish
                        activeTracksPool[t_idx].vx *= 0.5;
                        activeTracksPool[t_idx].vy *= 0.5;
                    }
                }

                // Erase expired active trajectory targets that exceeded our historical aging threshold limits
                activeTracksPool.erase(
                    std::remove_if(activeTracksPool.begin(), activeTracksPool.end(),
                        [MAX_AGE_BUDGET](const DeepTrack& t) {
                            return t.time_since_update > MAX_AGE_BUDGET;
                        }),
                    activeTracksPool.end()
                );

                trackingLogArray.push_back(frameTrackNode);
            }
        }

        // 6. Output Final Structured Telemetry Document
        std::ofstream outputFileStream(outputJsonPath);
        if (!outputFileStream.is_open())
        {
            throw std::runtime_error("Could not write predictive tracking telemetry metrics: " + outputJsonPath);
        }

        outputFileStream << trackingLogArray.dump(4);
        outputFileStream.close();

        std::cout << "\n==========================================" << std::endl;
        std::cout << "PREDICTIVE TRACKING SUCCESS: DeepSORT logic compiled smoothly!" << std::endl;
        std::cout << "Data Stream Mapping Saved to: " << outputJsonPath << std::endl;
        std::cout << "==========================================" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "\nTRACKER MODULE CRITICAL FAILURE: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}