#include "nms.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
class CollectCb : public NmsCb {
public:
    std::vector<BoundingBox> picked;

    int callback(BoundingBox &boundingBox) override
    {
        picked.push_back(boundingBox);
        return 0;
    }
};

std::string serialize(const std::vector<BoundingBox> &boxes)
{
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto &b = boxes[i];
        out << "{\"classId\":" << b.classId
            << ",\"score\":" << b.score
            << ",\"minX\":" << b.minX
            << ",\"minY\":" << b.minY
            << ",\"maxX\":" << b.maxX
            << ",\"maxY\":" << b.maxY
            << "}";
        if (i + 1 < boxes.size()) {
            out << ",";
        }
    }
    out << "]";
    return out.str();
}

std::string runCase1()
{
    NmsPostProcess nms;

    BoundingBox a(0, 0, 9, 9, 90, 0);
    BoundingBox b(1, 1, 10, 10, 80, 0);
    BoundingBox c(20, 20, 30, 30, 70, 0);
    nms.AddBoundingBox(a);
    nms.AddBoundingBox(b);
    nms.AddBoundingBox(c);

    CollectCb cb;
    nms.Go(50, cb);
    return serialize(cb.picked);
}

std::string runCase2()
{
    NmsPostProcess nms;

    BoundingBox c1_primary(2, 2, 6, 6, 85, 1);
    BoundingBox c2_same_box(2, 2, 6, 6, 75, 2);
    BoundingBox c1_overlap_low(2, 2, 5, 5, 50, 1);

    nms.AddBoundingBox(c1_primary);
    nms.AddBoundingBox(c2_same_box);
    nms.AddBoundingBox(c1_overlap_low);

    CollectCb cb;
    nms.Go(50, cb);
    return serialize(cb.picked);
}

std::string runNamedCase(const std::string &name)
{
    if (name == "nms_case1") {
        return runCase1();
    }
    if (name == "nms_case2") {
        return runCase2();
    }
    throw std::runtime_error("unknown case: " + name);
}

std::string readWholeFile(const std::filesystem::path &path)
{
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("cannot read golden file: " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string trimTrailingWhitespace(std::string input)
{
    while (!input.empty() &&
           (input.back() == '\n' || input.back() == '\r' || input.back() == ' ' || input.back() == '\t')) {
        input.pop_back();
    }
    return input;
}
}

int main(int argc, char **argv)
{
    if (argc == 3 && std::string(argv[1]) == "--dump") {
        std::cout << runNamedCase(argv[2]) << std::endl;
        return 0;
    }

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <golden_dir>" << std::endl;
        std::cerr << "   or: " << argv[0] << " --dump <nms_case1|nms_case2>" << std::endl;
        return 2;
    }

    const std::filesystem::path goldenDir(argv[1]);
    const std::vector<std::string> cases = {"nms_case1", "nms_case2"};

    bool failed = false;
    for (const auto &caseName : cases) {
        const std::string actual = trimTrailingWhitespace(runNamedCase(caseName));
        const std::string expected = trimTrailingWhitespace(readWholeFile(goldenDir / (caseName + ".json")));
        if (actual != expected) {
            failed = true;
            std::cerr << "Golden mismatch for " << caseName << std::endl;
            std::cerr << "Expected: " << expected << std::endl;
            std::cerr << "Actual  : " << actual << std::endl;
        }
    }

    return failed ? 1 : 0;
}
