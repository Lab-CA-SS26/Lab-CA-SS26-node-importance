#pragma once

#include <map>
#include <string>
#include <vector>
#include <sstream>

class CommandLineParser {

public:
    CommandLineParser(int argc, char** argv) {
        for (int currentIndex = 1; currentIndex < argc; ++currentIndex) {
            if (argv[currentIndex][0] == '-') {
                // Strip all leading dashes
                int dashCount = 1;
                while (argv[currentIndex][dashCount] == '-') {
                    dashCount++;
                }
                std::string key(&argv[currentIndex][dashCount]);
                
                std::string value("");
                // Only swallow the next token if it's not another flag
                if ((currentIndex + 1) < argc && argv[currentIndex + 1][0] != '-') {
                    value.assign(argv[currentIndex + 1]);
                    currentIndex++; 
                }
                arguments[key] = value;
            }
        }
    }

    template<typename T>
    inline T value(const std::string key, const T defaultValue = T()) const noexcept;

    template<typename T>
    inline T get(const std::string key, const T defaultValue = T()) const noexcept {
        return value<T>(key, defaultValue);
    }

    inline bool isSet(const std::string key) const noexcept {
        return arguments.find(key) != arguments.end();
    }

    inline size_t numberOfArguments() const noexcept {
        return arguments.size();
    }

private:
    std::map<std::string, std::string> arguments;

};

template<>
inline std::string CommandLineParser::value<std::string>(const std::string key, const std::string defaultValue) const noexcept {
    auto it = arguments.find(key);
    if (it != arguments.end()) {
        return it->second;
    }
    return defaultValue;
}

template<>
inline int CommandLineParser::value<int>(const std::string key, const int defaultValue) const noexcept {
    auto it = arguments.find(key);
    if (it != arguments.end() && !it->second.empty()) {
        try { return std::stoi(it->second); } catch(...) {}
    }
    return defaultValue;
}

template<>
inline double CommandLineParser::value<double>(const std::string key, const double defaultValue) const noexcept {
    auto it = arguments.find(key);
    if (it != arguments.end() && !it->second.empty()) {
        try { return std::stod(it->second); } catch(...) {}
    }
    return defaultValue;
}

template<>
inline bool CommandLineParser::value<bool>(const std::string key, const bool defaultValue) const noexcept {
    auto it = arguments.find(key);
    if (it != arguments.end()) {
        if (it->second.empty() || it->second == "true" || it->second == "1") return true;
        if (it->second == "false" || it->second == "0") return false;
        return true;
    }
    return defaultValue;
}
