#include "cache.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

TuneCache::TuneCache(
    std::string path
)
    :
    path_(
        std::move(path)
    ) {
    load();
}

void TuneCache::load() {
    records_.clear();

    std::ifstream in(path_);

    if (!in) {
        return;
    }

    std::string line;
    std::getline(in, line);

    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string field;

        CacheRecord r;

        std::getline(ss, r.gpu, ',');

        std::getline(ss, field, ',');
        r.cc_major = std::stoi(field);

        std::getline(ss, field, ',');
        r.cc_minor = std::stoi(field);

        std::getline(ss, r.dtype, ',');

        std::getline(ss, field, ',');
        r.M = std::stoi(field);

        std::getline(ss, field, ',');
        r.N = std::stoi(field);

        std::getline(ss, field, ',');
        r.K = std::stoi(field);

        std::getline(ss, field, ',');
        r.epilogue = std::stoi(field);

        std::getline(ss, r.kernel, ',');

        std::getline(ss, field, ',');
        r.latency_ms = std::stof(field);

        std::getline(ss, field, ',');
        r.tflops = std::stod(field);

        records_.push_back(
            std::move(r)
        );
    }
}

bool TuneCache::lookup(
    const std::string& gpu,
    int cc_major,
    int cc_minor,
    const std::string& dtype,
    int M,
    int N,
    int K,
    int epilogue,
    CacheRecord& out
) const {
    for (const auto& r : records_) {
        if (
            r.gpu == gpu
            &&
            r.cc_major == cc_major
            &&
            r.cc_minor == cc_minor
            &&
            r.dtype == dtype
            &&
            r.M == M
            &&
            r.N == N
            &&
            r.K == K
            &&
            r.epilogue == epilogue
        ) {
            out = r;
            return true;
        }
    }

    return false;
}

void TuneCache::upsert(
    const CacheRecord& record
) {
    for (auto& r : records_) {
        if (
            r.gpu == record.gpu
            &&
            r.cc_major == record.cc_major
            &&
            r.cc_minor == record.cc_minor
            &&
            r.dtype == record.dtype
            &&
            r.M == record.M
            &&
            r.N == record.N
            &&
            r.K == record.K
            &&
            r.epilogue == record.epilogue
        ) {
            r = record;
            return;
        }
    }

    records_.push_back(record);
}

void TuneCache::save() const {
    const std::filesystem::path p(path_);

    if (p.has_parent_path()) {
        std::filesystem::create_directories(
            p.parent_path()
        );
    }

    std::ofstream out(path_);

    if (!out) {
        throw std::runtime_error(
            "Failed to open tune cache: "
            +
            path_
        );
    }

    out
        << "gpu,cc_major,cc_minor,dtype,M,N,K,epilogue,kernel,latency_ms,tflops\n";

    out << std::setprecision(9);

    for (const auto& r : records_) {
        out
            << r.gpu << ","
            << r.cc_major << ","
            << r.cc_minor << ","
            << r.dtype << ","
            << r.M << ","
            << r.N << ","
            << r.K << ","
            << r.epilogue << ","
            << r.kernel << ","
            << r.latency_ms << ","
            << r.tflops
            << "\n";
    }
}
