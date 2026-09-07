#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <sqlite3.h>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

void executeSQL(sqlite3* db, const std::string& sql) {
    char* errorMessage = nullptr;
    int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &errorMessage);
    if (rc != SQLITE_OK) {
        std::string err = "SQL Error: " + std::string(errorMessage);
        sqlite3_free(errorMessage);
        throw std::runtime_error(err);
    }
}

int main() {
    sqlite3* db = nullptr;
    try {
        // POINT TO THE ML INTELLIGENCE OUTPUT FILE Manifest
        std::string jsonPath = "detections/smart_collection.json";
        if (!fs::exists(jsonPath)) {
            throw std::runtime_error("ML Smart collection data missing! Run data_collector first. Path: " + jsonPath);
        }

        std::cout << "Database Ingesting ML Optimized Data From: " << jsonPath << std::endl;
        std::ifstream inputFile(jsonPath);
        json smartData;
        inputFile >> smartData;
        inputFile.close();

        std::string dbDir = "src/database";
        std::string dbPath = dbDir + "/drone_telemetry.db";

        if (fs::exists(dbPath)) {
            fs::remove(dbPath);
        }

        int rc = sqlite3_open(dbPath.c_str(), &db);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("Cannot initialize SQL database: " + std::string(sqlite3_errmsg(db)));
        }

        std::string createDetectionTable = 
            "CREATE TABLE IF NOT EXISTS detection_table ("
            "detection_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "frame_name TEXT UNIQUE NOT NULL,"
            "video_token TEXT NOT NULL"
            ");";

        std::string createObjectTable = 
            "CREATE TABLE IF NOT EXISTS object_table ("
            "object_entry_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "detection_id INTEGER NOT NULL,"
            "class TEXT NOT NULL,"
            "confidence REAL NOT NULL,"
            "x1 REAL NOT NULL,"
            "y1 REAL NOT NULL,"
            "x2 REAL NOT NULL,"
            "y2 REAL NOT NULL,"
            "FOREIGN KEY(detection_id) REFERENCES detection_table(detection_id)"
            ");";

        std::string createTimelineTable = 
            "CREATE TABLE IF NOT EXISTS timeline ("
            "timeline_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "detection_id INTEGER NOT NULL,"
            "timestamp_marker TEXT DEFAULT CURRENT_TIMESTAMP,"
            "FOREIGN KEY(detection_id) REFERENCES detection_table(detection_id)"
            ");";

        // UPDATED SCHEMA: Appended fields to track ML intelligence properties natively
        std::string createTrackingTable = 
            "CREATE TABLE IF NOT EXISTS tracking ("
            "track_entry_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "detection_id INTEGER NOT NULL,"
            "track_id INTEGER NOT NULL,"
            "class TEXT NOT NULL,"
            "value_score REAL NOT NULL,"
            "collection_rationale TEXT NOT NULL,"
            "FOREIGN KEY(detection_id) REFERENCES detection_table(detection_id)"
            ");";

        executeSQL(db, createDetectionTable);
        executeSQL(db, createObjectTable);
        executeSQL(db, createTimelineTable);
        executeSQL(db, createTrackingTable);

        executeSQL(db, "BEGIN TRANSACTION;");

        std::string insertDetectionSQL = "INSERT OR IGNORE INTO detection_table (frame_name, video_token) VALUES (?, ?);";
        std::string insertObjectSQL    = "INSERT INTO object_table (detection_id, class, confidence, x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?, ?, ?);";
        std::string insertTimelineSQL  = "INSERT INTO timeline (detection_id) VALUES (?);";
        std::string insertTrackingSQL  = "INSERT INTO tracking (detection_id, track_id, class, value_score, collection_rationale) VALUES (?, ?, ?, ?, ?);";

        sqlite3_stmt* stmtDetection = nullptr;
        sqlite3_stmt* stmtObject    = nullptr;
        sqlite3_stmt* stmtTimeline  = nullptr;
        sqlite3_stmt* stmtTracking  = nullptr;

        sqlite3_prepare_v2(db, insertDetectionSQL.c_str(), -1, &stmtDetection, nullptr);
        sqlite3_prepare_v2(db, insertObjectSQL.c_str(), -1, &stmtObject, nullptr);
        sqlite3_prepare_v2(db, insertTimelineSQL.c_str(), -1, &stmtTimeline, nullptr);
        sqlite3_prepare_v2(db, insertTrackingSQL.c_str(), -1, &stmtTracking, nullptr);

        int frameCount = 0;
        int entityCount = 0;

        for (const auto& frameEntry : smartData) {
            std::string frameName = frameEntry.value("frame", "");
            if (frameName.empty()) continue;

            sqlite3_bind_text(stmtDetection, 1, frameName.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_text(stmtDetection, 2, "test_02", -1, SQLITE_STATIC);
            sqlite3_step(stmtDetection);
            sqlite3_reset(stmtDetection);
            frameCount++;

            sqlite3_int64 detectionId = sqlite3_last_insert_rowid(db);

            sqlite3_bind_int64(stmtTimeline, 1, detectionId);
            sqlite3_step(stmtTimeline);
            sqlite3_reset(stmtTimeline);

            if (frameEntry.contains("collected_targets") && frameEntry["collected_targets"].is_array()) {
                for (const auto& target : frameEntry["collected_targets"]) {
                    int trackId = target.value("track_id", -1);
                    std::string className = target.value("class", "unknown");
                    double confidence = target.value("confidence", 0.0);
                    double x1 = target.value("x1", 0.0);
                    double y1 = target.value("y1", 0.0);
                    double x2 = target.value("x2", 0.0);
                    double y2 = target.value("y2", 0.0);

                    // Pull nested intelligence details
                    double mlScore = 0.0;
                    std::string mlRationale = "Collected.";
                    if (target.contains("ml_metrics")) {
                        mlScore = target["ml_metrics"].value("value_score", 0.0);
                        mlRationale = target["ml_metrics"].value("rationale", "Collected.");
                    }

                    sqlite3_bind_int64(stmtObject, 1, detectionId);
                    sqlite3_bind_text(stmtObject, 2, className.c_str(), -1, SQLITE_STATIC);
                    sqlite3_bind_double(stmtObject, 3, confidence);
                    sqlite3_bind_double(stmtObject, 4, x1);
                    sqlite3_bind_double(stmtObject, 5, y1);
                    sqlite3_bind_double(stmtObject, 6, x2);
                    sqlite3_bind_double(stmtObject, 7, y2);
                    sqlite3_step(stmtObject);
                    sqlite3_reset(stmtObject);
                    entityCount++;

                    // Bind custom metrics directly to database schema columns
                    sqlite3_bind_int64(stmtTracking, 1, detectionId);
                    sqlite3_bind_int(stmtTracking, 2, trackId);
                    sqlite3_bind_text(stmtTracking, 3, className.c_str(), -1, SQLITE_STATIC);
                    sqlite3_bind_double(stmtTracking, 4, mlScore);
                    sqlite3_bind_text(stmtTracking, 5, mlRationale.c_str(), -1, SQLITE_STATIC);
                    sqlite3_step(stmtTracking);
                    sqlite3_reset(stmtTracking);
                }
            }
        }

        sqlite3_finalize(stmtDetection);
        sqlite3_finalize(stmtObject);
        sqlite3_finalize(stmtTimeline);
        sqlite3_finalize(stmtTracking);

        executeSQL(db, "COMMIT;");
        std::cout << "Relational compilation active. Ingested " << frameCount << " critical event frames." << std::endl;
        sqlite3_close(db);
    }
    catch (const std::exception& e) {
        if (db) { sqlite3_close(db); }
        std::cout << "\nError: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}