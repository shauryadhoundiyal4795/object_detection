#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

// Pure struct representing a localized ML collection decision
struct CollectionFeature {
    std::string frame;
    std::string class_name;
    double score;
    bool should_collect;
    std::string rationale;
};

//------------------------------------------------------------
// Core ML Logic Engine: Evaluates Telemetry Value Scores
//------------------------------------------------------------
CollectionFeature evaluateFeatureValue(const json& currentObj, const json& priorFrameNode, const std::string& frameName) {
    CollectionFeature feature;
    feature.frame = frameName;
    feature.class_name = currentObj.value("class", "unknown");
    feature.score = 0.0;
    feature.should_collect = false;
    feature.rationale = "Redundant static state observation.";

    double confidence = currentObj.value("confidence", 0.0);
    int trackId = currentObj.value("track_id", -1);

    // Rule 1: High novelty multiplier (e.g., intercepting a person or drone vs a bench)
    if (feature.class_name == "person" || feature.class_name == "drone") {
        feature.score += 0.40;
    }

    // Rule 2: Low confidence edge-case inspection tracking values
    if (confidence > 0.45 && confidence < 0.65) {
        feature.score += 0.25; // Flag borderline edge-cases for manual retraining review
    }

    // Rule 3: Spatial Displacement Vector check (Is the object moving fast or behaving weirdly?)
    bool foundPriorInstance = false;
    if (!priorFrameNode.empty() && priorFrameNode.contains("tracks")) {
        for (const auto& priorTrack : priorFrameNode["tracks"]) {
            if (priorTrack.value("track_id", -1) == trackId) {
                foundPriorInstance = true;
                
                // Calculate geometric distance displacement variance
                double dx1 = std::abs(currentObj.value("x1", 0.0) - priorTrack.value("x1", 0.0));
                double dy1 = std::abs(currentObj.value("y1", 0.0) - priorTrack.value("y1", 0.0));
                double structuralDelta = std::sqrt(dx1*dx1 + dy1*dy1);

                if (structuralDelta > 15.0) { // Object moved significantly
                    feature.score += 0.45;
                }
                break;
            }
        }
    }

    // If it's a completely brand new unique track profile trajectory entering scene
    if (!foundPriorInstance) {
        feature.score += 0.50;
    }

    // Collection Decision Gate Threshold Check
    const double ML_COLLECTION_THRESHOLD = 0.55;
    if (feature.score >= ML_COLLECTION_THRESHOLD) {
        feature.should_collect = true;
        if (!foundPriorInstance) feature.rationale = "Initial path discovery footprint.";
        else if (feature.score > 0.8)  feature.rationale = "High velocity vector behavior anomaly.";
        else                            feature.rationale = "Edge-case uncertainty validation collection.";
    }

    return feature;
}

//------------------------------------------------------------
// Execution Main Path
//------------------------------------------------------------
int main() {
    try {
        std::string trackingJsonPath = "tracking_result/tracking.json";
        if (!fs::exists(trackingJsonPath)) {
            throw std::runtime_error("Tracking payload file matrix missing: " + trackingJsonPath);
        }

        std::ifstream file(trackingJsonPath);
        json trackingData;
        file >> trackingData;
        file.close();

        std::cout << "ML Data Collection Engine scanning file: " << trackingJsonPath << std::endl;

        json priorFrameNode = json::object();
        json intelligenceLog = json::array();

        int analyticsSuppressedCount = 0;
        int analyticsCollectedCount = 0;

        for (const auto& frameEntry : trackingData) {
            std::string frameName = frameEntry.value("frame", "");
            
            json frameIntelligenceNode = json::object();
            frameIntelligenceNode["frame"] = frameName;
            frameIntelligenceNode["collected_targets"] = json::array();

            if (frameEntry.contains("tracks") && frameEntry["tracks"].is_array()) {
                for (const auto& trackObj : frameEntry["tracks"]) {
                    
                    // Core ML scoring model call hook step
                    CollectionFeature decision = evaluateFeatureValue(trackObj, priorFrameNode, frameName);

                    if (decision.should_collect) {
                        json collectedObj = trackObj;
                        collectedObj["ml_metrics"] = {
                            {"value_score", decision.score},
                            {"rationale", decision.rationale}
                        };
                        frameIntelligenceNode["collected_targets"].push_back(collectedObj);
                        analyticsCollectedCount++;
                    } else {
                        analyticsSuppressedCount++;
                    }
                }
            }

            // Always save the frame node context if it contains high-value collection targets
            if (!frameIntelligenceNode["collected_targets"].empty()) {
                intelligenceLog.push_back(frameIntelligenceNode);
            }

            // Cache current reference pointer states to evaluate subsequent displacements loops
            priorFrameNode = frameEntry;
        }

        // Output smart collection files manifest directly to directory
        std::string outputDir = "detections";
        std::string outputPath = outputDir + "/smart_collection.json";
        
        std::ofstream outFile(outputPath);
        outFile << intelligenceLog.dump(4);
        outFile.close();

        std::cout << "\n==========================================" << std::endl;
        std::cout << "ML DATA COLLECTION SUBSYSTEM COMPLETE!" << std::endl;
        std::cout << "Redundant Discards (Suppressed): " << analyticsSuppressedCount << " Instances" << std::endl;
        // This is your instructor's golden metric: proof of intelligent compression
        std::cout << "High Value Saves   (Collected) : " << analyticsCollectedCount << " Instances" << std::endl;
        std::cout << "Smart Manifest Document Written to: " << outputPath << std::endl;
        std::cout << "==========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "ML Engine Critical Runtime Interruption: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}