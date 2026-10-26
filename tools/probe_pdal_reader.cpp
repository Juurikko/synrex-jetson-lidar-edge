// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tuomas Pietilä
// Optional diagnostic linked to external PDAL, outside native inference.
#include <iostream>
#include <pdal/Options.hpp>
#include <pdal/PDALUtils.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/QuickInfo.hpp>
#include <pdal/StageFactory.hpp>
#include <string>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr
            << "usage: probe_pdal_reader FILE preview|prepare|prepare-preview|preview-preview\n";
        return 2;
    }
    try {
        pdal::StageFactory factory;
        auto *reader = factory.createStage("readers.las");
        if (!reader)
            return 3;
        pdal::Options options;
        options.add("filename", argv[1]);
        reader->setOptions(options);
        pdal::PointTable table;
        const std::string mode(argv[2]);
        if (mode == "prepare" || mode == "prepare-preview") {
            reader->prepare(table);
            std::cout << "prepare PASS\n";
        }
        if (mode == "preview-preview") {
            const auto first = reader->preview();
            std::cout << "first preview PASS " << first.m_pointCount << "\n";
        }
        if (mode == "preview" || mode == "prepare-preview" || mode == "preview-preview") {
            const auto info = reader->preview();
            std::cout << "preview PASS " << info.m_pointCount << "\n";
        } else if (mode != "prepare") {
            return 2;
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
