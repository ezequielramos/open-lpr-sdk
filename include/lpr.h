#pragma once

#include <string>

struct LprResult {
    std::string plate;
    float confidence;
};

class LprEngine {

public:
    LprEngine();

    LprResult process(const unsigned char* frame,
                      int width,
                      int height);

};
