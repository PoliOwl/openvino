// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "SubgraphCutter.hpp"
//#include "SubgraphCutter.cpp"



//struct for file paths used to get needed data
struct FilePath {

std::string IRpath;
std::string namePath;
std::string weightsPath = "";
std::string inputPath;
std::string forVisualPath;
std::string inputFileName = "Result";
std::string deviceName =  "MYRIAD";
std::string helpMessage = "Help message: \n\tUsage: vpu_graph_editor <model IR path> [<weights path>] <txt file with nodes name path> [-i <image path>] [-f <inputs file name>] [-c]\n\timage MUST be bmp 24bpp\n\t swithc -c is used for calculating subgraph and comparing resuts\n";
bool calcNet = false;
//void defult{};

//gets values from command line 
FilePath(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Missing required command line arguments" << std::endl;
        std::cout << helpMessage;
        std::exit(EXIT_FAILURE);
    }
    if (argc == 3) {
       IRpath = argv[1];
       namePath = argv[2];
        for(const auto& latter : IRpath){
                if(latter == '.') {
                    break;
                }
                weightsPath += latter;
            }
        weightsPath+= ".bin";
    }
    else {
        int begin = 0;
        if(argv[3][0] !='-') {
        IRpath = argv[1];
        namePath = argv[3];
        weightsPath = argv[2];
        begin = 4;
        }
        else {
            IRpath = argv[1];
            namePath = argv[2];
            for(const auto& latter : IRpath){
                if(latter == '.') {
                    break;
                }
                weightsPath += latter;
            }
            weightsPath+= ".bin";
            begin = 3;
        }
        for(int i = begin; i < argc; ++i) {
            std::string operation = argv[i];
            if(operation == "-i") {
                inputPath = argv[++i];
                continue;
            }
            if(operation == "-f") {
                inputFileName = argv[++i];
                continue;
            }
            if(operation == "-c") {
                calcNet = true;
                continue;
            }
            std::cout<<"Unknown operaion " << argv[i]<<std::endl;
            std::cout << helpMessage;
            std::exit(EXIT_FAILURE);
        }
    }
}

};


//reads names from file
void readNames(std::set<std::string>& NamesCont, const std::string& filePath) {
    std::fstream f;
    f.open(filePath, std::fstream::in);
    if(!f.is_open()) {
        std::cout << "\nfiled to open file\n";
        std::exit(EXIT_FAILURE);
    }
    std::string s; 
    while (std::getline(f, s)) {
       if(s.size() > 0){
           NamesCont.insert(std::move(s));
       }
    }
    f.close();
}



int main(int argc, char* argv[]) {
    FilePath file(argc, argv);
    std::set<std::string> names; //set of subgraphs nodes names
    readNames(names, file.namePath);
    SubgraphCutter subCut(file.IRpath, file.weightsPath, names);
    std::cout<< "Subgraph created\n";
    if(file.inputPath.size()) {
        std::map<std::string, std::string> saveFileNames;
        const auto& inputs = subCut.getInputs(file.inputPath, file.inputFileName, saveFileNames);
        std::cout << "input file names: \n";
        for(auto& save : saveFileNames) {
            std::cout<<"\tfor "<<save.first<<"  "<<save.second<<"\n";
        }
        if(file.calcNet) {
            const auto& comparationResults = subCut.compareResults(saveFileNames, inputs); 
            std::cout<<"Comparing results:\n";
            for(auto& res : comparationResults) {
                std::cout<<"\t"<<res.first<<" max diff: "<<res.second<<"\n";
            }
        }
    }
    return 0;
   
}
