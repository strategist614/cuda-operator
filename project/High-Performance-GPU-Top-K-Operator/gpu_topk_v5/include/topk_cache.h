#pragma once

#include <string>
#include <vector>

struct TopKCacheRecord {
    std::string gpu;
    int cc_major = 0;
    int cc_minor = 0;

    std::string dtype;

    int batch = 0;
    int n = 0;
    int k = 0;

    std::string kernel;

    float latency_ms = 0.0f;
    double gelem_per_s = 0.0;
};

class TopKTuneCache {
public:
    explicit TopKTuneCache(std::string path);

    bool lookup(
        const std::string& gpu,
        int cc_major,
        int cc_minor,
        const std::string& dtype,
        int batch,
        int n,
        int k,
        TopKCacheRecord& out
    ) const;

    void upsert(
        const TopKCacheRecord& record
    );

    void save() const;

    const std::string& path() const {
        return path_;
    }

private:
    void load();

    std::string path_;
    std::vector<TopKCacheRecord> records_;
};
