#pragma once

#include <string>

struct CropRatios {
    float x_left = 0.0f;
    float x_right = 0.0f;
    float y_top = 0.0f;
    float y_bottom = 0.0f;
};

bool parseCropRatio(const std::string &input, float &value);

bool computeCropRect(int image_width,
                     int image_height,
                     const CropRatios &ratios,
                     int &x,
                     int &y,
                     int &width,
                     int &height,
                     std::string &error);
