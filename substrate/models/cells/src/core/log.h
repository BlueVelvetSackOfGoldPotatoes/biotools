#pragma once

#include <fstream>
#include <string>

namespace cells {

class CsvLogger {
public:
    CsvLogger() = default;

    bool open(const std::string& path, const std::string& header) {
        out_.open(path, std::ios::out | std::ios::trunc);
        if (!out_.is_open()) {
            return false;
        }
        out_ << header << '\n';
        out_.flush();
        return true;
    }

    void write_row(const std::string& row) {
        if (!out_.is_open()) {
            return;
        }
        out_ << row << '\n';
    }

    void flush() {
        if (out_.is_open()) {
            out_.flush();
        }
    }

private:
    std::ofstream out_;
};

} // namespace cells

