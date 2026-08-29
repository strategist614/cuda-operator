#pragma once

#include "gemm.h"

#include <string>
#include <vector>

struct CacheRecord {
    std::string gpu;
    int cc_major = 0;
    int cc_minor = 0;

    std::string dtype;

    int M = 0;
    int N = 0;
    int K = 0;

    int epilogue = 0;

    std::string kernel;

    float latency_ms = 0.0f;
    double tflops = 0.0;
};

class TuneCache {
public:
    explicit TuneCache(std::string path);

    bool lookup(
        const std::string& gpu,
        int cc_major,
        int cc_minor,
        const std::string& dtype,
        int M,
        int N,
        int K,
        int epilogue,
        CacheRecord& out
    ) const;

    void upsert(
        const CacheRecord& record
    );

    void save() const;

    const std::string& path() const {
        return path_;
    }

private:
    void load();

    std::string path_;
    std::vector<CacheRecord> records_;
};
