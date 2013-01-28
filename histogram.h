#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>
#include <iostream>

class Histogram {

protected:
    std::vector<unsigned> *hist;
    //unsigned* hist;
    int num_bins;
    int delta;
    int min;
    unsigned sum;

public:
    Histogram(unsigned int bins, int smin, int smax) {
        hist = new std::vector<unsigned>();
        hist->resize(bins, 0);
        //hist = new unsigned[bins];
        //for(int i=0; i < bins; i++) {
            //hist[bins]=0;
        //}
        num_bins = bins;
        min = smin;
        sum = 0;
        delta = (smax-smin)/bins;
    }

    ~Histogram() {
        delete hist;
    }

    int chooseBin(int sample) {
        // shift to the origin, then divide into delta to figure out bin number
        int bin = (sample - min) / delta;
        // Handle samples above or below range
        if (bin < 0) bin = 0;
        if (bin >= num_bins) bin = num_bins-1;

        return bin;
    }

    void add(int sample) {
        int bin = chooseBin(sample);
        hist->at(bin)=hist->at(bin)+1;
        sum++;
    }

    void merge(Histogram* h) {
        // TODO: Assert num_bins, min, max ==
        for(unsigned int n=0; n<num_bins; n++) {
            hist->at(n) += h->hist->at(n);
            sum += h->hist->at(n);
        }
    }

    double chiSquared(Histogram* h) {
        double chi = 0.0;
        for (unsigned int i=0; i<num_bins; i++) {
            // Get bins and normalize them
            double ss = hist->at(i) / double(sum);
            double hs = h->hist->at(i) / double(h->sum);

            // Compute chi-squared
            double a = ss + hs;
            if (a == 0.0) continue;
            double b = ss - hs;
            chi += b*b / a;
        }
        return chi/2.0;
    }

    double intersection(Histogram* h) {
        unsigned sumofmins = 0;
        for (unsigned i=0; i<num_bins; i++) {
            sumofmins += std::min(hist->at(i), h->hist->at(i));
        }
        unsigned minofsums = std::min(sum, h->sum);
        return 1-(sumofmins/double(minofsums));
    }

    void print() {
        for (unsigned int i = 0; i < num_bins; i++) {
            std::cout << hist->at(i)/double(sum) << " ";
        }
        std::cout << std::endl;
    }
};

#endif
