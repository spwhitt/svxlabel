#ifndef VIDEO_H
#define VIDEO_H

#include <dirent.h>
#include <string>
#include <iostream>
#include <opencv/highgui.h>

using namespace std;

vector<string>* list_files(string dir) {
    DIR* dir_stream = opendir(dir.c_str());

    if (!dir_stream) {
        cout << "Failed to open directory " << dir << endl;
        exit(EXIT_FAILURE);
    }

    dirent* entry;

    vector<string>* paths = new vector<string>();

    // Get image paths
    while ((entry = readdir(dir_stream))) {
        // If the entry is a regular file
        if (entry->d_type == DT_REG) {
            // Get the filename
            string fname = entry->d_name;
            fname = dir + "/" + fname;
            paths->push_back(fname);
        }
    }

    // clean up
    closedir(dir_stream);

    if (!paths->size()) {
        cout << "Directory is empty" << endl;
        exit(EXIT_FAILURE);
    }

    sort(paths->begin(), paths->end());

    return paths;
}

class SvSpace {
public:
    unsigned cols, rows, frames;
    vector<cv::Mat>* data;
    SvSpace() {
        data = new vector<cv::Mat>();
    }
    ~SvSpace() {
        delete data;
    }
    ushort getPixel(unsigned x, unsigned y, unsigned t) {
        return data->at(t).at<ushort>(y, x);
    }
    void setPixel(unsigned x, unsigned y, unsigned t, unsigned value) {
        data->at(t).at<ushort>(y, x) = value;
    }

    void write() {
        char buff[15];

        for (unsigned z = 0; z < data->size(); z++) {
            sprintf(buff, "output%04d.png", z);
            cv::imwrite(string(buff), data->at(z));
        }
    }
};

SvSpace* load_video_frames(string frame_dir) {

    SvSpace* svspace = new SvSpace();

    vector<string>* paths = list_files(frame_dir);

    svspace->data->reserve(paths->size());

    // Read each image file from list of paths
    for (unsigned i = 0; i < paths->size(); i++) {
        string fname = paths->at(i);
        cv::Mat im = cv::imread(fname, CV_LOAD_IMAGE_ANYDEPTH);

        if (!im.data) {
            cout << "Skipping non-image file " << fname << endl;
        } else {
            svspace->data->push_back(im);
        }
    }

    // TODO: More elegant way of discovering and cross-checking this information
    svspace->frames = paths->size();
    svspace->cols = svspace->data->at(0).cols;
    svspace->rows = svspace->data->at(0).rows;

    delete paths;

    return svspace;
}

#endif
