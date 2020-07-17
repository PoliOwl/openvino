#include <iostream>
#include <inference_engine.hpp>
#include <generic_ie.hpp>
#include "net_pass.h"
#include <fstream>
#include <set>
#include <string>
#include <cstdlib>
#include "cnn_network_ngraph_impl.hpp"


#define DEBUG 0


//struct for file paths used to get needed data
struct filePaths {

std::string IRpath;
std::string namePath;
std::string weightsPath;
//void defult{};

//gets values from command line 
filePaths(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Missing required command line arguments" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (argc == 3) {
       IRpath = argv[1];
       namePath = argv[2];
       weightsPath = "NONE";
    }
    else {
        IRpath = argv[1];
        namePath = argv[3];
        weightsPath = argv[2];
    }
}

};


//reads names from file
void readNames(std::set<std::string>& NamesCont, const std::string& filePath) {
    std::fstream f;
    f.open(filePath, std::fstream::in);
    std::string s; 
    while (std::getline(f, s)) {
       if(s.size() > 0){
           NamesCont.insert(s);
       }
    }
}


//reads network
InferenceEngine::details::CNNNetworkNGraphImpl* getNetwork(const filePaths& file) {
    InferenceEngine::CNNNetwork network;
    if (file.weightsPath != "NONE") {
        network = InferenceEngine::Core().ReadNetwork(file.IRpath, file.weightsPath);
    }
    else {
        network = InferenceEngine::Core().ReadNetwork(file.IRpath);
    }
    InferenceEngine::ICNNNetwork::Ptr ptr = static_cast<InferenceEngine::ICNNNetwork::Ptr>(network);
   
    auto ngraphNetwork = dynamic_cast<InferenceEngine::details::CNNNetworkNGraphImpl*>(ptr.get());
    return ngraphNetwork;
}

int main(int argc, char* argv[]) {
    filePaths file(argc, argv);
    std::set<std::string> names;
    readNames(names, file.namePath);
    InferenceEngine::CNNNetwork network;
    if (file.weightsPath != "NONE") {
        network = InferenceEngine::Core().ReadNetwork(file.IRpath, file.weightsPath);
    }
    else {
        network = InferenceEngine::Core().ReadNetwork(file.IRpath);
    }
    InferenceEngine::ICNNNetwork::Ptr ptr = static_cast<InferenceEngine::ICNNNetwork::Ptr>(network);
    auto ngraphNetwork = dynamic_cast<InferenceEngine::details::CNNNetworkNGraphImpl*>(ptr.get()); 
    if (ngraphNetwork == nullptr) {
        std::cout << "failed to read network" << std::endl;
        return 1;
    }
    auto nGraphFunc = ngraphNetwork->getFunction();
    auto ops = nGraphFunc->get_ordered_ops();
    std::fstream f;
    std::set<std::string> check = names;
    for (auto op : ops) {
        check.erase(op->get_friendly_name());
        if (check.empty()) {
            break;
        }
    }
    if (!check.empty()) {
        for (auto name : check) {
            std::cout << name <<std::endl;
        }
        std::cout << "wasn't found. Stop"<< std::endl;
        return 0;
    }
    std::cout << "All names are valid" << std::endl;
    return 0;
}
