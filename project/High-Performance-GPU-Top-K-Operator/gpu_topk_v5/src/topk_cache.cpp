#include "topk_cache.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

TopKTuneCache::TopKTuneCache(
    std::string path
)
    :
    path_(
        std::move(path)
    ) {
    load();
}

void TopKTuneCache::load() {
    records_.clear();

    std::ifstream in(path_);

    if (!in) {
        return;
    }

    std::string line;

    /*
     * header
     */
    std::getline(
        in,
        line
    );

    while (
        std::getline(
            in,
            line
        )
    ) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string field;

        TopKCacheRecord r;

        std::getline(
            ss,
            r.gpu,
            ','
        );

        std::getline(
            ss,
            field,
            ','
        );
        r.cc_major =
            std::stoi(field);

        std::getline(
            ss,
            field,
            ','
        );
        r.cc_minor =
            std::stoi(field);

        std::getline(
            ss,
            r.dtype,
            ','
        );

        std::getline(
            ss,
            field,
            ','
        );
        r.batch =
            std::stoi(field);

        std::getline(
            ss,
            field,
            ','
        );
        r.n =
            std::stoi(field);

        std::getline(
            ss,
            field,
            ','
        );
        r.k =
            std::stoi(field);

        std::getline(
            ss,
            r.kernel,
            ','
        );

        std::getline(
            ss,
            field,
            ','
        );
        r.latency_ms =
            std::stof(field);

        std::getline(
            ss,
            field,
            ','
        );
        r.gelem_per_s =
            std::stod(field);

        records_.push_back(
            std::move(r)
        );
    }
}

bool TopKTuneCache::lookup(
    const std::string& gpu,
    int cc_major,
    int cc_minor,
    const std::string& dtype,
    int batch,
    int n,
    int k,
    TopKCacheRecord& out
) const {
    for (
        const auto& r :
        records_
    ) {
        if (
            r.gpu == gpu
            &&
            r.cc_major == cc_major
            &&
            r.cc_minor == cc_minor
            &&
            r.dtype == dtype
            &&
            r.batch == batch
            &&
            r.n == n
            &&
            r.k == k
        ) {
            out =
                r;

            return true;
        }
    }

    return false;
}

void TopKTuneCache::upsert(
    const TopKCacheRecord& record
) {
    for (
        auto& r :
        records_
    ) {
        if (
            r.gpu == record.gpu
            &&
            r.cc_major == record.cc_major
            &&
            r.cc_minor == record.cc_minor
            &&
            r.dtype == record.dtype
            &&
            r.batch == record.batch
            &&
            r.n == record.n
            &&
            r.k == record.k
        ) {
            r =
                record;

            return;
        }
    }

    records_.push_back(
        record
    );
}

void TopKTuneCache::save() const {
    const std::filesystem::path p(
        path_
    );

    if (
        p.has_parent_path()
    ) {
        std::filesystem::create_directories(
            p.parent_path()
        );
    }

    std::ofstream out(
        path_
    );

    if (!out) {
        throw std::runtime_error(
            "Failed to open Top-K tune cache: "
            +
            path_
        );
    }

    out
        << "gpu,cc_major,cc_minor,dtype,batch,n,k,kernel,latency_ms,gelem_per_s\n";

    out
        << std::setprecision(9);

    for (
        const auto& r :
        records_
    ) {
        out
            << r.gpu << ","
            << r.cc_major << ","
            << r.cc_minor << ","
            << r.dtype << ","
            << r.batch << ","
            << r.n << ","
            << r.k << ","
            << r.kernel << ","
            << r.latency_ms << ","
            << r.gelem_per_s
            << "\n";
    }
}
