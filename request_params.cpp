#include "request_params.h"

#include <cmath>
#include <stdexcept>

namespace {
constexpr float kMinRatio = 0.0f;
constexpr float kMaxRatio = 1.0f;
}

bool parseCropRatio(const std::string &input, float &value)
{
    try {
        size_t parsed = 0;
        float parsedValue = std::stof(input, &parsed);
        if (parsed != input.size()) {
            return false;
        }
        if (!std::isfinite(parsedValue)) {
            return false;
        }
        if (parsedValue < kMinRatio || parsedValue > kMaxRatio) {
            return false;
        }
        value = parsedValue;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

bool computeCropRect(int image_width,
                     int image_height,
                     const CropRatios &ratios,
                     int &x,
                     int &y,
                     int &width,
                     int &height,
                     std::string &error)
{
    if (image_width <= 0 || image_height <= 0) {
        error = "error: invalid image size";
        return false;
    }

    if (ratios.x_left + ratios.x_right >= 1.0f ||
        ratios.y_top + ratios.y_bottom >= 1.0f) {
        error = "error: invalid crop ratios";
        return false;
    }

    x = static_cast<int>(image_width * ratios.x_left);
    y = static_cast<int>(image_height * ratios.y_top);

    width = static_cast<int>(image_width * (1.0f - ratios.x_left - ratios.x_right));
    height = static_cast<int>(image_height * (1.0f - ratios.y_top - ratios.y_bottom));

    if ((width % 2) != 0) {
        --width;
    }
    if ((height % 2) != 0) {
        --height;
    }

    if (width <= 0 || height <= 0) {
        error = "error: crop area is empty";
        return false;
    }

    if (x < 0 || y < 0 || x + width > image_width || y + height > image_height) {
        error = "error: crop area out of image bounds";
        return false;
    }

    return true;
}
