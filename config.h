#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

void parse_config_file(string file_path, string* frames_dir, string* supervoxel_dir, string* gtruth_path) {
    ifstream config_file(file_path);
    config_file >> *frames_dir;
    config_file >> *supervoxel_dir;
    config_file >> *gtruth_path;
}

#endif
