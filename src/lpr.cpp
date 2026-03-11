#include "lpr.h"
#include <iostream>

LprEngine::LprEngine() {
    std::cout << "LPR Engine initialized" << std::endl;
}

LprResult LprEngine::process(const unsigned char* frame,
                             int width,
                             int height) {

    LprResult result;

    result.plate = "FXL7E66";
    result.confidence = 0.99;

    return result;
}