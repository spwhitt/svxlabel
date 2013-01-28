#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <dirent.h>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
//#include <tr1/unordered_map>
#include "histogram.h"

//
// Future Ideas:
// create a "video source" object, which gets passed to svxinfo
// There could be a video source which loads the frames dynamically,
// or one which has them all preloaded into memory.
//

//
// Standardized names:
// sv: a single supervoxel
// svimg: an image of supervoxel labels (2d matrix)
// svspace: 3d space of supervoxel labels
// svimg_path: file location of a svimg
//
// vid: 
// vidframe: 
// 

using namespace std;

unsigned NUM_HIST_BINS = 10;
unsigned NUM_SUPERVOXELS = 35987;
unsigned NUM_LABELS = 14;

struct Sv;

vector<string>* list_files(string dir) {
    DIR* dir_stream = opendir(dir.c_str());
    if (!dir_stream) {
        cout << "Failed to open directory " << dir << endl;
        exit(EXIT_FAILURE);
    }

    dirent* entry;

    vector<string>* paths = new vector<string>();
    // Get image paths
    while( (entry = readdir(dir_stream)) ) {
        // If the entry is a regular file
        if (entry->d_type == DT_REG){ 
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

struct Link {
    double weight;
    Sv *begin, *end;
};

struct Sv {
    unsigned index, size;
    unsigned first, last;
    int label;
    Histogram *L, *a, *b;
    set<unsigned>* neighbors;
    vector<Link*>* fwd_links;

    Sv() : index(0), size(0), first(0), last(0), label(-1) {
        L = new Histogram(NUM_HIST_BINS, 0, 255);
        a = new Histogram(NUM_HIST_BINS, 0, 255);
        b = new Histogram(NUM_HIST_BINS, 0, 255);
        neighbors = new set<unsigned>();
        fwd_links = new vector<Link*>();
    }
    ~Sv() {
        delete L;
        delete a;
        delete b;
        delete neighbors;
        delete fwd_links;
    }
};

vector<Sv*>* svxinfo(string sv_dir, string video_dir) {

    //
    // Part 1: Load each video frame, and process each pixel one at a time
    // extract the following information for each supervoxel
    //  - Size in pixels
    //  - Starting frame
    //  - Ending frame
    //  - Neighbors
    //  - CIE-Lab histograms
    //

    vector<string>* svimg_paths = list_files(sv_dir);
    vector<string>* video_paths = list_files(video_dir);

    unsigned numframes = svimg_paths->size();
    // TODO: Assert that supervoxel_paths and video_paths have the same number of frames, same size...

    // TODO: Fix the sizing of this array to something more reasonable
    // a hardcoded # of supervoxels is painful, and just asking for trouble...
    //Sv svs[NUM_SUPERVOXELS];
    vector<Sv*>* svs = new vector<Sv*>(NUM_SUPERVOXELS);
    //unordered_map<unsigned, Sv*> svs

    cv::Mat svimg_prev;

    for(unsigned z=0; z < numframes; z++) {
        // Load input images
        cv::Mat svimg = cv::imread(svimg_paths->at(z), CV_LOAD_IMAGE_ANYDEPTH);
        cv::Mat frame = cv::imread(video_paths->at(z));
        cv::cvtColor(frame, frame, CV_BGR2Lab);

        for(unsigned y=0; y < (unsigned)svimg.rows; y++) {
            for(unsigned x=0; x < (unsigned)svimg.cols; x++) {
                // Get color and supervoxel label for the current x,y,z location
                ushort svimg_pix = svimg.at<ushort>(y, x);
                cv::Vec3b frame_pix = frame.at<cv::Vec3b>(y, x);

                // compute supervoxel size and location
                // TODO: Try, catch around this line, to resize vector if out-of-bounds
                Sv* sv = svs->at(svimg_pix);
                if(!sv) {
                    sv = new Sv();
                    svs->at(svimg_pix) = sv;
                    sv->index = svimg_pix;
                    sv->first = z;
                }

                sv->size++;
                sv->last=z;

                // build histograms
                sv->L->add(frame_pix[0]);
                sv->a->add(frame_pix[1]);
                sv->b->add(frame_pix[2]);

                // find neighboring supervoxels
                ushort neigh;
                if(x > 0){
                    neigh = svimg.at<ushort>(y, x-1);
                    sv->neighbors->insert(neigh);
                }
                if(y > 0){
                    neigh = svimg.at<ushort>(y-1, x);
                    sv->neighbors->insert(neigh);
                }
                if(z > 0){
                    neigh = svimg_prev.at<ushort>(y, x);
                    sv->neighbors->insert(neigh);
                }
            }
        }

        // Hold on to the previous frame's supervoxels, so we can
        // check for neighboring supervoxels in the temporal dimension
        svimg_prev = svimg;
    }

    delete svimg_paths;
    delete video_paths;

    //
    // Part 2: Process each supervoxel
    // create the fwd and rvs links, as well as the weight of each one
    //

    for (unsigned i=0; i < NUM_SUPERVOXELS; i++) {
        if(svs->at(i)) {
            Sv* sv = svs->at(i);

            for(set<unsigned>::iterator it = sv->neighbors->begin(); it!=sv->neighbors->end(); it++) {
                Sv* neigh = svs->at(*it);

                if(sv->first < neigh->first) {
                    Link* link = new Link();

                    // TODO: Ask Albert about chiSquared, possibly switch back to intersection or something
                    double chisq_L =  sv->L->chiSquared(neigh->L);
                    double chisq_a =  sv->a->chiSquared(neigh->a);
                    double chisq_b =  sv->b->chiSquared(neigh->b);
                    double weight = (1-chisq_L) * (1-chisq_a) * (1-chisq_b);

                    link->begin = sv;
                    link->end = neigh;
                    link->weight = weight;

                    // Add the forward links to the neighbor instead of the original supervoxel
                    // The neighbor needs to know about incoming links when deciding which label to choose
                    // The other supervoxel doesn't really care...
                    neigh->fwd_links->push_back(link);
                }

                // Reverse Links
                //if(sv->last > neigh->last) {
                //rvs_links.push_back(neigh->index);
                //}
            }
        }
    }

    return svs;
}

// Sort by which supervoxel 
bool firstInitialFrame(Sv* sv1, Sv* sv2) {
    // Have to handle the ugly case of sv1, or sv2 being an null pointer
    // This is a great argument for switching to a hashmap or something...
    if (!sv1) {
        return false;
    } else if (!sv2) {
        return true;
        // Simply return whichever one has the earlier first frame
    } else {
        return sv1->first < sv2->first;
    }
}

bool greatestWeight(Link* l1, Link* l2) {
    return l1->weight > l2->weight;
}

int main(int argc, char** argv) {

    //cv::Mat lut(1, NUM_SUPERVOXELS, CV_8U);

    // Get iterators
    // Get information about the supervoxels...
    vector<Sv*>* svs = svxinfo("/vpml-scratch/spencer/data/bus/swa/05/", "/vpml-scratch/spencer/data/bus/frames/");

    // Get the first frame to label a few initial supervoxels
    cv::Mat gtruth = cv::imread("/vpml-scratch/spencer/data/bus/labels/0001.png", CV_LOAD_IMAGE_GRAYSCALE); // 8 bit
    cv::Mat fstsvs = cv::imread("/vpml-scratch/spencer/data/bus/swa/05/0001.png", CV_LOAD_IMAGE_ANYDEPTH);  // 16 bit

    cv::MatIterator_<uchar> gtruth_itr = gtruth.begin<uchar>();
    cv::MatIterator_<ushort> fstsvs_itr = fstsvs.begin<ushort>();

    // Iterate through every pixel
    for (; gtruth_itr != gtruth.end<uchar>(); gtruth_itr++, fstsvs_itr++) {
        // Naive implementation: simply takes the last pixel pair
        // TODO: Reimplement to perform mode; get the most frequent label under the supervoxel
        ushort svlabel = *fstsvs_itr;
        Sv* sv = svs->at(svlabel);
        if(sv) {
            sv->label = (int)*gtruth_itr;
        }
    }

    // Sort by initial frame
    sort( svs->begin(), svs->end(), firstInitialFrame );

    ofstream myfile;
    myfile.open ("out.csv");

    int count = 0;
    // Propagate the labels along the fwd links
    for(vector<Sv*>::iterator i=svs->begin(); i != svs->end(); i++) {
        if(*i) {
            vector<Link*>* fwd = (*i)->fwd_links;

            // Alread has a label, no need to compute a new one
            if ((*i)->label != -1) {
                continue;
            }

            if(fwd->size() == 0) {
                count ++;
                cout << (*i)->index << endl;
            } else {
                Link* strongest = *max_element(fwd->begin(), fwd->end(), greatestWeight);

                Sv* choice = strongest->begin;

                // Assign label
                (*i)->label = choice->label;
            }

            myfile << (*i)->index << ", " << (*i)->label << endl;
        } else {
            cout << "skip" << endl;
        }
    }

    cout << count << " svs have no incoming links " << endl;

    myfile.close();

    delete svs;

    return 0;
}
