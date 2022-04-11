// #pragma once

// #include <rapidjson/document.h>
// #include <rapidjson/filewritestream.h>
// #include <rapidjson/prettywriter.h>
// #include <rapidjson/writer.h>
// #include <rapidjson/stringbuffer.h>

// #include <fstream>

// // Use RapidJSON to write a log file.
// struct JSONLog {
//   JSONLog(const std::string& _filename) : filename(_filename) {
//     // Set up a JSON object
//     alloc = document.GetAllocator();
//     document.SetObject();
//   }

//   // Single key-value pair
//   // void write(const std::string& key, const std::string& value) {
//   //   rapidjson::Value k(key.c_str(), alloc);
//   //   rapidjson::Value v(value.c_str(), alloc);
//   //   document.AddMember(k, v, alloc);
//   // }

//   // Key-array pair

//   // Write the log to disk

//  private:
//   std::string filename;
//   rapidjson::Document document;
//   rapidjson::Document::AllocatorType alloc;
// };