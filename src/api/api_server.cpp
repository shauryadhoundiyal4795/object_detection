#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <sqlite3.h>
#include <crow.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

//------------------------------------------------------------
// Database Helper: Reads detection/object table
//------------------------------------------------------------
json getTelemetryDataFromDB(const std::string& dbPath) {
    sqlite3* db = nullptr;
    json rootArray = json::array();

    int rc = sqlite3_open(dbPath.c_str(), &db);
    if (rc != SQLITE_OK) {
        std::cerr << "SQL Error: Unable to read database file." << std::endl;
        return rootArray;
    }

    std::string query = 
        "SELECT d.frame_name, d.video_token, o.class, o.confidence, o.x1, o.y1, o.x2, o.y2 "
        "FROM detection_table d "
        "LEFT JOIN object_table o ON d.detection_id = o.detection_id "
        "ORDER BY d.detection_id ASC;";

    sqlite3_stmt* stmt = nullptr;
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
    
    if (rc == SQLITE_OK) {
        std::string currentFrame = "";
        json currentFrameNode;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* frameNameRaw = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            const char* videoTokenRaw = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            const char* classTxt = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));

            std::string frameName = frameNameRaw ? frameNameRaw : "";
            std::string videoToken = videoTokenRaw ? videoTokenRaw : "";
            std::string className = classTxt ? classTxt : "";

            if (frameName.empty()) continue;

            if (frameName != currentFrame) {
                if (!currentFrame.empty()) {
                    rootArray.push_back(currentFrameNode);
                }
                currentFrame = frameName;
                currentFrameNode = json::object();
                currentFrameNode["frame"] = frameName;
                currentFrameNode["video_token"] = videoToken;
                currentFrameNode["objects"] = json::array();
            }

            if (!className.empty()) {
                json obj;
                obj["class"] = className;
                obj["confidence"] = sqlite3_column_double(stmt, 3);
                obj["bbox"] = {
                    {"x1", sqlite3_column_double(stmt, 4)},
                    {"y1", sqlite3_column_double(stmt, 5)},
                    {"x2", sqlite3_column_double(stmt, 6)},
                    {"y2", sqlite3_column_double(stmt, 7)}
                };
                currentFrameNode["objects"].push_back(obj);
            }
        }
        if (!currentFrame.empty()) {
            rootArray.push_back(currentFrameNode);
        }
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return rootArray;
}

//------------------------------------------------------------
// Database Helper: Reads tracking analytics and ML insights
//------------------------------------------------------------
json getTrackingDataFromDB(const std::string& dbPath) {
    sqlite3* db = nullptr;
    json rootArray = json::array();

    int rc = sqlite3_open(dbPath.c_str(), &db);
    if (rc != SQLITE_OK) {
        std::cerr << "SQL Error: Unable to read database file." << std::endl;
        return rootArray;
    }

    // UPDATED SQL QUERY: Selecting new intelligence metric indicators
    std::string query = 
        "SELECT d.frame_name, t.track_id, t.class, o.confidence, o.x1, o.y1, t.value_score, t.collection_rationale "
        "FROM detection_table d "
        "JOIN tracking t ON d.detection_id = t.detection_id "
        "LEFT JOIN object_table o ON d.detection_id = o.detection_id AND t.class = o.class "
        "GROUP BY d.frame_name, t.track_id, t.class "
        "ORDER BY d.detection_id ASC;";

    sqlite3_stmt* stmt = nullptr;
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
    
    if (rc == SQLITE_OK) {
        std::string currentFrame = "";
        json currentFrameNode;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* frameNameRaw = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            int trackId = sqlite3_column_int(stmt, 1);
            const char* classTxt = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            double confidence = sqlite3_column_double(stmt, 3);
            double x1 = sqlite3_column_double(stmt, 4);
            double y1 = sqlite3_column_double(stmt, 5);
            double valueScore = sqlite3_column_double(stmt, 6);
            const char* rationaleRaw = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));

            std::string frameName = frameNameRaw ? frameNameRaw : "";
            std::string className = classTxt ? classTxt : "";
            std::string rationale = rationaleRaw ? rationaleRaw : "Collected.";

            if (frameName.empty()) continue;

            if (frameName != currentFrame) {
                if (!currentFrame.empty()) {
                    rootArray.push_back(currentFrameNode);
                }
                currentFrame = frameName;
                currentFrameNode = json::object();
                currentFrameNode["frame"] = frameName;
                currentFrameNode["tracks"] = json::array();
            }

            json trackItem;
            trackItem["track_id"] = trackId;
            trackItem["class"] = className;
            trackItem["confidence"] = confidence;
            trackItem["x1"] = x1;
            trackItem["y1"] = y1;
            trackItem["ml_value_score"] = valueScore;
            trackItem["ml_rationale"] = rationale;
            
            currentFrameNode["tracks"].push_back(trackItem);
        }
        if (!currentFrame.empty()) {
            rootArray.push_back(currentFrameNode);
        }
    } else {
        std::cerr << "SQL Tracking Query Error: " << sqlite3_errmsg(db) << std::endl;
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return rootArray;
}

//------------------------------------------------------------
// Main Routing Service Execution Loop
//------------------------------------------------------------
int main() {
    crow::SimpleApp app;
    std::string dbLocation = "src/database/drone_telemetry.db";

    CROW_ROUTE(app, "/")([](){
        return "Drone Interceptor AI Pipeline Core REST Engine Running Online.";
    });

    CROW_ROUTE(app, "/api/detections")([&dbLocation](){
        json dataPayload = getTelemetryDataFromDB(dbLocation);
        crow::response res;
        res.set_header("Content-Type", "application/json");
        res.set_header("Access-Control-Allow-Origin", "*"); 
        res.body = dataPayload.dump(4);
        return res;
    });

    CROW_ROUTE(app, "/api/tracks")([&dbLocation](){
        json dataPayload = getTrackingDataFromDB(dbLocation);
        crow::response res;
        res.set_header("Content-Type", "application/json");
        res.set_header("Access-Control-Allow-Origin", "*"); 
        res.body = dataPayload.dump(4);
        return res;
    });

    CROW_ROUTE(app, "/images/<string>")([](std::string filename){
        crow::response res;
        std::string imagePath = "tracked_frames/" + filename;
        res.set_header("Access-Control-Allow-Origin", "*"); 
        if (std::filesystem::exists(imagePath)) {
            res.set_static_file_info(imagePath);
            res.set_header("Content-Type", "image/jpeg");
        } else {
            res.code = 404;
            res.body = "Tracked image file matrix placeholder missing: " + imagePath;
        }
        return res;
    });

    std::cout << "\n==========================================" << std::endl;
    std::cout << "API SERVER SERVICE ACTIVE: All routes upgraded successfully!" << std::endl;
    std::cout << "Detections Endpoint : http://localhost:8080/api/detections" << std::endl;
    std::cout << "Tracking Endpoint   : http://localhost:8080/api/tracks" << std::endl;
    std::cout << "==========================================" << std::endl;

    app.port(8080).multithreaded().run();
    return 0;
}