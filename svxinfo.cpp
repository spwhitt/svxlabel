#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <dirent.h>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <unordered_map>
#include "histogram.h"
#include "video.h"

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

#define NUM_HIST_BINS 10
#define NUM_LABELS 24

struct Sv;

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

// Type which associates pixel values with Sv structures
typedef unordered_map<unsigned, Sv*> SvMap;

SvMap* svxinfo(SvSpace* svspace, string video_dir) {

    //
    // Part 1: Load each video frame, and process each pixel one at a time
    // extract the following information for each supervoxel
    //  - Size in pixels
    //  - Starting frame
    //  - Ending frame
    //  - Neighbors
    //  - CIE-Lab histograms
    //

    vector<string>* video_paths = list_files(video_dir);

    // TODO: Assert that supervoxel_paths and video_paths have the same number of frames, same size...
    // TODO: Fix the sizing of this array to something more reasonable
    // a hardcoded # of supervoxels is painful, and just asking for trouble...
    // TODO: Speed up Mat access with row pointers
    //

    SvMap* svs = new SvMap();

    for(unsigned z=0; z < svspace->frames; z++) {
        // Load input frame
        cv::Mat frame = cv::imread(video_paths->at(z));
        cv::cvtColor(frame, frame, CV_BGR2Lab);

        for(unsigned y=0; y < (unsigned)svspace->rows; y++) {
            for(unsigned x=0; x < (unsigned)svspace->cols; x++) {
                // Get color and supervoxel label for the current x,y,z location
                ushort svimg_pix = svspace->getPixel(x, y, z);
                cv::Vec3b frame_pix = frame.at<cv::Vec3b>(y, x);

                // compute supervoxel size and location
                Sv* sv;
                try {
                    sv = svs->at(svimg_pix);
                } catch (out_of_range& oor) {
                    sv = new Sv();
                    svs->insert({svimg_pix, sv});
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
                    neigh = svspace->getPixel(x-1, y, z);
                    sv->neighbors->insert(neigh);
                    // TODO: Think up a way to make the following line less ugly
                    svs->at(neigh)->neighbors->insert(sv->index);
                }
                if(y > 0){
                    neigh = svspace->getPixel(x, y-1, z);
                    sv->neighbors->insert(neigh);
                    svs->at(neigh)->neighbors->insert(sv->index);
                }
                if(z > 0){
                    neigh = svspace->getPixel(x, y, z-1);
                    sv->neighbors->insert(neigh);
                    svs->at(neigh)->neighbors->insert(sv->index);
                }
            }
        }
    }

    delete video_paths;

    //
    // Part 2: Process each supervoxel
    // create the fwd and rvs links, as well as the weight of each one
    //

    for(auto it=svs->begin(); it != svs->end(); it++) {
        Sv* sv = it->second;

        for(set<unsigned>::iterator it = sv->neighbors->begin(); it!=sv->neighbors->end(); it++) {
            Sv* neigh = svs->at(*it);

            if(sv->first < neigh->first) {
                Link* link = new Link();
                double intersect_L =  sv->L->intersection(neigh->L);
                double intersect_a =  sv->a->intersection(neigh->a);
                double intersect_b =  sv->b->intersection(neigh->b);
                double weight = (1-intersect_L) * (1-intersect_a) * (1-intersect_b);

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

    return svs;
}

// Sort by which supervoxel 
bool cmpInitialFrame(Sv* sv1, Sv* sv2) {
    // Have to handle the ugly case of sv1, or sv2 being an null pointer
    // This is a great argument for switching to a hashmap or something...
    if (!sv1) {
        return false;
    } else if (!sv2) {
        return true;
    } else {
        // Simply return whichever one has the earlier first frame
        return sv1->first < sv2->first;
    }
}

bool cmpWeight(Link* l1, Link* l2) {
    return l1->weight < l2->weight;
}

int main(int argc, char** argv) {

    cout << "Start." << endl;

    double time = (double)cv::getTickCount();

    // 
    // Process command line arguments
    //
    string frames_dir;
    string supervoxel_dir;
    string gtruth_path;
    if(argc == 4) {
        frames_dir = argv[1];
        supervoxel_dir = argv[2];
        gtruth_path = argv[3];
    } else if (argc==1) {
        cout << "No arguments specified, using default bus sequence" << endl;
        frames_dir = "/vpml-scratch/spencer/data/bus/frames/";
        supervoxel_dir = "/vpml-scratch/spencer/data/bus/swa/05/";
        gtruth_path = "/vpml-scratch/spencer/data/bus/labels/0001.png";
    } else {
        cout << "Invalid arguments." << endl;
        exit(EXIT_FAILURE);
    }

    SvSpace* svspace = load_video_frames(supervoxel_dir);

    // Get information about the supervoxels...
    SvMap* svs = svxinfo(svspace, frames_dir);

    // Ground truth and supervoxels from the first frame
    cv::Mat gtruth = cv::imread(gtruth_path, CV_LOAD_IMAGE_GRAYSCALE); // 8 bit
    cv::Mat fstsvs = svspace->data->at(0);

    cv::MatIterator_<uchar> gtruth_itr = gtruth.begin<uchar>();
    cv::MatIterator_<ushort> fstsvs_itr = fstsvs.begin<ushort>();

    // Compute the mode label for each supervoxel on the first frame
    unordered_map<unsigned, vector<unsigned>> mode;
    // Iterate through every pixel
    for (; gtruth_itr != gtruth.end<uchar>(); gtruth_itr++, fstsvs_itr++) {
        ushort svlabel = *fstsvs_itr;
        int label = *gtruth_itr;
        // TODO: The resize below is ugly, there is definitely a better way
        mode[svlabel].resize(NUM_LABELS);
        mode[svlabel][label]++;
        Sv* sv = svs->at(svlabel);
        if(mode[svlabel][label] > mode[svlabel][sv->label]) {
            sv->label = label;
        }
    }

    // Create a copy of the svs vector, this one sorted by initial frame
    vector<Sv*> svsSortedFF;
    svsSortedFF.reserve(svs->size());
    for(auto it=svs->begin(); it != svs->end(); it++) {
        svsSortedFF.push_back(it->second);
    }
    sort( svsSortedFF.begin(), svsSortedFF.end(), cmpInitialFrame );

    int count = 0;
    // Propagate the labels along the fwd links
    for(vector<Sv*>::iterator i=svsSortedFF.begin(); i != svsSortedFF.end(); i++) {
        vector<Link*>* fwd = (*i)->fwd_links;

        // Alread has a label, no need to compute a new one
        if ((*i)->label != -1) {
            continue;
        }

        if(fwd->size() == 0) {
            count++;
        } else {
            Link* strongest = *max_element(fwd->begin(), fwd->end(), cmpWeight);

            Sv* choice = strongest->begin;

            // Assign label
            (*i)->label = choice->label;
        }
    }

    cout << count << " svs have no incoming links " << endl;

    //
    // Visualize results!!!!!
    //

    for(unsigned z=0; z < svspace->frames; z++) {
        for(unsigned y=0; y < (unsigned)svspace->rows; y++) {
            for(unsigned x=0; x < (unsigned)svspace->cols; x++) {
                ushort supervoxel = svspace->getPixel(x,y,z);
                int value = svs->at(supervoxel)->label;
                // TODO: figure out WHY some values are -1, when logically they should not...
                // Also, perhaps implement appearance model here
                if (value == -1) {
                    value = 0;
                }
                svspace->setPixel(x, y, z, value);
            }
        }
    }

    // TODO: Convert to 8 bit?
    svspace->write();

    time = ((double)cv::getTickCount() - time)/cv::getTickFrequency();
    cout << "Time passed: " << time << " seconds" << endl;

    delete svs;
    delete svspace;

    return 0;
}
