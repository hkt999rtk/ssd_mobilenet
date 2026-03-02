#include "request_params.h"

#include <cmath>
#include <iostream>

namespace {
int failures = 0;

void expect(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        ++failures;
    }
}
}

int main()
{
    float value = -1.0f;
    expect(parseCropRatio("0.25", value), "parseCropRatio should parse normal value");
    expect(std::fabs(value - 0.25f) < 1e-6f, "parseCropRatio should keep parsed value");

    expect(!parseCropRatio("abc", value), "parseCropRatio should reject non-numeric");
    expect(!parseCropRatio("1.2", value), "parseCropRatio should reject value > 1");
    expect(!parseCropRatio("-0.2", value), "parseCropRatio should reject value < 0");
    expect(!parseCropRatio("0.1x", value), "parseCropRatio should reject trailing chars");

    CropRatios okRatios;
    okRatios.x_left = 0.125f;
    okRatios.x_right = 0.125f;
    okRatios.y_top = 0.25f;
    okRatios.y_bottom = 0.25f;

    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    std::string error;
    expect(computeCropRect(1920, 1080, okRatios, x, y, width, height, error),
           "computeCropRect should accept valid ratios");
    expect(x == 240, "crop x should match expected value");
    expect(y == 270, "crop y should match expected value");
    expect(width == 1440, "crop width should match expected value");
    expect(height == 540, "crop height should match expected value");

    CropRatios badRatios;
    badRatios.x_left = 0.6f;
    badRatios.x_right = 0.4f;
    expect(!computeCropRect(100, 100, badRatios, x, y, width, height, error),
           "computeCropRect should reject horizontal sum >= 1");

    CropRatios emptyRatios;
    expect(!computeCropRect(1, 1, emptyRatios, x, y, width, height, error),
           "computeCropRect should reject empty effective crop after even rounding");

    if (failures > 0) {
        std::cerr << "Total failures: " << failures << std::endl;
        return 1;
    }
    return 0;
}
