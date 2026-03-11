#include "lpr.h"
#include <iostream>

int main() {

    std::cout << "Starting LPR test..." << std::endl;

    LprEngine engine;

    unsigned char frame[10]; // mock frame

    auto result = engine.process(frame, 100, 100);

    std::cout << "Plate: " << result.plate << std::endl;
    std::cout << "Confidence: " << result.confidence << std::endl;

    return 0;
}
