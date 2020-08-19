// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pass/visualize_tree.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "net_pass.h"
#include "ngraph_functions/pass/convert_prc.hpp"
#include "../common/vpu_tools_common.cpp"

#include <ngraph/opsets/opset3.hpp>
#include <inference_engine.hpp>
#include <generic_ie.hpp>
#include <common.hpp>
#include <vpu/vpu_plugin_config.hpp>
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include <iostream>
#include <fstream>
#include <set>
#include <vector>


#include <string>
#include <cstdlib>
#include <typeindex>
#include <map>



void readNames(std::set<std::string>&, const std::string&); //read subgraph nodes names from file
bool connectionCheck(std::shared_ptr<ngraph::Function>&, const  std::set<std::string>& ); //checks if graph is complete. If it is, return "", if not - name of node that is unnable to rich from every node
void visit(const std::shared_ptr<ngraph::Node>&, std::map<std::string, bool>&); 
std::shared_ptr<ngraph::Node> make_before_op(std::map<std::string, std::shared_ptr<ngraph::Node>>& before, std::shared_ptr<ngraph::Node>& op, ngraph::ParameterVector& parameters); //creates node for beforegraph
std::shared_ptr<ngraph::Node> createOp(std::map<std::string, std::shared_ptr<ngraph::Node>>& nodes, std::shared_ptr<ngraph::Node>& op, 
                                        ngraph::ParameterVector& parameters, ngraph::ResultVector& results, std::map<std::string, std::shared_ptr<ngraph::Node>>& before, 
                                        ngraph::ParameterVector& beforeParameters, ngraph::ResultVector& beforeResults, bool add_node_to_before); //creats operation, 
//if add_node_to_before = false, then it doesn't uses before.. parametrs  and only creates node, otherwise invokes make_before_op using before and before beforeParameters and adds result to beforeResults 
size_t findIndex(const ngraph::ResultVector&,const std::string&); //finds index of a result, that has input with given friendly name
std::vector<std::uint8_t> loadImage(const std::string &,std::shared_ptr<ngraph::opset3::Parameter>&); //loads data for parameter for given file




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



std::vector<std::uint8_t> loadImage(const std::string &imageFilename,std::shared_ptr<ngraph::opset3::Parameter>& param) {
    //auto& tens = param->get_output_tensor(0);
    //auto layout = tens.get_tensor_layout();

    BitMap reader(imageFilename);

    const auto dims = param->get_output_shape(0);

    const size_t N = dims[0];
    const size_t C = dims[1];
    const size_t H = dims[2];
    const size_t W = dims[3];

    const size_t img_w = reader.width();
    const size_t img_h = reader.height();

    const size_t numImageChannels = reader.size() / (reader.width() * reader.height());
    if (C != numImageChannels && C != 1) {
        std::cout << "loadImage error: Input channels mismatch: image channels " << numImageChannels << ", "
                  << "network channels " << C << ", expecting count of image channels are equal "
                  << "to count if network channels or count of network channels are equal to 1" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<std::uint8_t> info; 
    const unsigned char* RGB8 = reader.getData().get();
    const float xScale = 1.0f * img_w / W;
    const float yScale = 1.0f * img_h / H;

    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; ++h) {
            int y = static_cast<int>(std::floor((h + 0.5f) * yScale));
            for (int w = 0; w < W; ++w) {
                int x = static_cast<int>(std::floor((w + 0.5f) * xScale));
                for (int c = 0; c < C; c++) {
                    float tmp = 0.f;
                    tmp = 1.0 * RGB8[(y * img_w + x) * numImageChannels + c];
                    auto tmpVec = reinterpret_cast<std::uint8_t*>(&tmp);
                    for(int i = 0; i < sizeof(float); ++i ) {
                        info.push_back(tmpVec[i]);
                    }
                }
            }
        }
    }

    return info;
}


size_t findIndex(const ngraph::ParameterVector& results,const std::string& name) {
    size_t i = 0;
    for(auto res : results) {
        if(res->get_friendly_name() == name) {
            return i;
        }
        ++i;
    }
    return results.size();
}

size_t findIndex(const ngraph::ResultVector& results,const std::string& name) {
    size_t i = 0;
    for(auto res : results) {
        if(res->get_input_node_shared_ptr(0)->get_friendly_name() == name) {
            return i;
        }
        ++i;
    }
    return results.size();
}

template <typename T>
std::shared_ptr<T> find(std::vector<std::shared_ptr<T>>& vector,const std::string& node) { 
    for(auto el : vector) {
        if(el->get_friendly_name() == node) {
            return el;
        }
    }
    return std::shared_ptr<T>(nullptr);
}



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

void visit(const std::shared_ptr<ngraph::Node>& node, std::map<std::string, bool>& map) {
    map[node->get_friendly_name()] = true;
    for(auto& inp : node->input_values()) {
        const auto& input = inp.get_node_shared_ptr();
        if(map[input->get_friendly_name()] == false) {
            visit(input, map);
        }
    }
    if(strcmp(node->get_type_info().name, "Result") != 0) {
        for(size_t i = 0; i < node->get_output_size();++i) {
            const auto& set = node->get_output_target_inputs(i); 
            for (auto& sel : set) {
                auto el = sel.get_node();
                if(map[el->get_friendly_name()] == false) {
                    const auto& elPtr = el->output(0).get_node_shared_ptr();
                    ///auto ptr = std::make_shared<ngraph::Node>(*el);
                    visit(elPtr, map);
                }
            }
        }
    }
    
}

std::string connectionCheck(const std::shared_ptr<ngraph::Function>& func,const  std::set<std::string>& originalNames) {
    const auto& ops = func->get_ordered_ops();
    std::map<std::string, bool> visited;
    for(auto op : ops) {
        visited[op->get_friendly_name()] = false;
    }
    visit(ops[0], visited);
    for(auto el : visited) {
        if(el.second == false) {
            if(originalNames.count(el.first)) {
                return el.first;
            }
        }
    }
    return "";
}

std::shared_ptr<ngraph::Node> make_before_op(std::map<std::string, std::shared_ptr<ngraph::Node>>& before,const std::shared_ptr<ngraph::Node>& op, ngraph::ParameterVector& parameters) {
    if(strcmp(op->get_type_info().name, "Parameter") == 0) {
        const auto& stubParameter = std::make_shared<ngraph::opset3::Parameter>(op->get_element_type(), op->get_shape());
        stubParameter->set_friendly_name(op->get_friendly_name());
        parameters.push_back(stubParameter);
        return stubParameter;
    }
    ngraph::OutputVector args;
    for(auto& input : op->input_values()) {
        const auto& el = input.get_node_shared_ptr();
        auto befNode = before.find(el->get_friendly_name());
        if(befNode != before.end()) {
            args.push_back(befNode->second);
            continue;
        }
        auto existingParameter = find(parameters, el->get_friendly_name());
        if(existingParameter.get() != nullptr) {
            args.push_back(existingParameter);
            continue;
        }
        if(strcmp(el->get_type_info().name, "Constant") == 0) {
            const auto& constant= dynamic_cast<ngraph::opset3::Constant*>(el.get());
            const auto& subConstant = std::make_shared<ngraph::opset3::Constant>(constant->get_element_type(), constant->get_shape(), constant->get_value_strings());
            subConstant->set_friendly_name(el->get_friendly_name());
            before[el->get_friendly_name()] = subConstant;
            args.push_back(subConstant);
            continue;
        }
        if(strcmp(op->get_type_info().name, "Parameter") == 0) {
            const auto& stubParameter = std::make_shared<ngraph::opset3::Parameter>(input.get_element_type(), input.get_partial_shape());
            stubParameter->set_friendly_name(el->get_friendly_name());
            parameters.push_back(stubParameter);
            args.push_back(stubParameter);
            continue;
        }
        args.push_back(make_before_op(before, el, parameters));
    }
    auto newOp = op->clone_with_new_inputs(args);
    newOp->set_friendly_name(op->get_friendly_name());
    before[op->get_friendly_name()] = newOp;
    return newOp;
}

std::shared_ptr<ngraph::Node> createOp(std::map<std::string, std::shared_ptr<ngraph::Node>>& nodes,const std::shared_ptr<ngraph::Node>& op, 
                                        ngraph::ParameterVector& parameters,ngraph::ResultVector& results, std::map<std::string, std::shared_ptr<ngraph::Node>>& before, 
                                        ngraph::ParameterVector& beforeParameters, ngraph::ResultVector& beforeResults, bool addNodeToBefore = false){
     std::shared_ptr<ngraph::Node> newOp(nullptr);
            if(strcmp(op->get_type_info().name, "Parameter") != 0) {
                ngraph::OutputVector args;
                for(auto& element : op->input_values()) {
                    const auto& el = element.get_node_shared_ptr(); 
                    const auto& name = el->get_friendly_name();
                    auto subNode = nodes.find(name);
                    if(subNode != nodes.end()) {
                        args.push_back(subNode->second);
                        continue;
                    }
                    const auto& existingParameter = find(parameters, name);
                    if(existingParameter.get() != nullptr) {
                        args.push_back(existingParameter);
                        continue;
                    }
                    if(strcmp(el->get_type_info().name, "Constant") == 0) {
                        const auto& constant= dynamic_cast<ngraph::opset3::Constant*>(el.get());
                        const auto& subConstant = std::make_shared<ngraph::opset3::Constant>(constant->get_element_type(), constant->get_shape(), constant->get_value_strings());
                        subConstant->set_friendly_name(el->get_friendly_name());
                        nodes[el->get_friendly_name()] = subConstant;
                        args.push_back(subConstant);
                        continue;
                    }
                    const auto& stubParameter = std::make_shared<ngraph::opset3::Parameter>(element.get_element_type(), element.get_partial_shape());
                    stubParameter->set_friendly_name(name);
                    parameters.push_back(stubParameter);
                    args.push_back(stubParameter);
                    if(addNodeToBefore && (strcmp(el->get_type_info().name, "Parameter") != 0)) {
                        auto before_node = make_before_op(before, el, beforeParameters);
                        beforeResults.push_back(std::make_shared<ngraph::opset3::Result>(before_node));
                    }
                }
                if(strcmp(op->get_type_info().name, "Result") == 0) {
                    const auto& newRes = std::make_shared<ngraph::opset3::Result>(args[0].get_node_shared_ptr());
                    newRes->set_friendly_name(op->get_friendly_name());
                    results.push_back(newRes);
                } else {
                    newOp = op->clone_with_new_inputs(args);
                    newOp->set_friendly_name(op->get_friendly_name());
                    nodes[op->get_friendly_name()] = newOp;
                }
            }
            else {
                const auto& stubParameter = std::make_shared<ngraph::opset3::Parameter>(op->get_element_type(), op->get_shape());
                stubParameter->set_friendly_name(op->get_friendly_name());
                newOp = stubParameter;
                parameters.push_back(stubParameter);
            }
    return newOp;
}



void loadBinaryTensorf32(const std::string &binaryFileName, InferenceEngine::Blob::Ptr& blob) {
    InferenceEngine::TensorDesc tensDesc = blob->getTensorDesc();
    if (tensDesc.getPrecision() != InferenceEngine::Precision::FP16) {
        throw std::invalid_argument("Input must have FP16 precision");
    }

    std::ifstream binaryFile(binaryFileName, std::ios_base::binary | std::ios_base::ate);
    if (!binaryFile) {
        throw std::invalid_argument("Can not open \"" + binaryFileName + "\"");
    }

    auto fileSize = static_cast<std::size_t>(binaryFile.tellg());
    binaryFile.seekg(0, std::ios_base::beg);
    if (!binaryFile.good()) {
        throw std::invalid_argument("Can not read \"" + binaryFileName + "\"");
    }
    /* try to read 32 bits data */
    std::int16_t *blobDataPtr = std::dynamic_pointer_cast<InferenceEngine::TBlob<std::int16_t>>(blob)->data();
    for (std::size_t i = 0; i < blob->size(); i++) {
        float tmp = 0.f;
        binaryFile.read(reinterpret_cast<char *>(&tmp), sizeof(float));
        blobDataPtr[i] = InferenceEngine::PrecisionUtils::f32tof16(tmp);
    }
}

bool compare(const std::vector<uint8_t>& ref, InferenceEngine::Blob::Ptr res) {
    auto refFloat = reinterpret_cast<const float*>(ref.data());
    short* blobDataPtr = std::dynamic_pointer_cast<InferenceEngine::TBlob<short>>(res)->data();
    for(auto i = 0; i < (ref.size()/sizeof(float)); ++i) {
        if(blobDataPtr[i] != InferenceEngine::PrecisionUtils::f32tof16(refFloat[i])) {
            return false;
        }
    }
    return true;
}


int main(int argc, char* argv[]) {
    FilePath file(argc, argv);
    std::set<std::string> names; //set of subgraphs nodes names
    readNames(names, file.namePath);
    InferenceEngine::CNNNetwork network;
    network = InferenceEngine::Core().ReadNetwork(file.IRpath, file.weightsPath);
    InferenceEngine::ICNNNetwork::Ptr ptr = static_cast<InferenceEngine::ICNNNetwork::Ptr>(network);
    auto ngraphNetwork = dynamic_cast<InferenceEngine::details::CNNNetworkNGraphImpl*>(ptr.get()); 
    if (ngraphNetwork == nullptr) {
        std::cout << "failed to read network" << std::endl;
        return 1;
    }
    const auto& nGraphFunc = ngraphNetwork->getFunction();
    const auto& ops = nGraphFunc->get_ordered_ops();
    ngraph::ResultVector subgraphResults;
    ngraph::ResultVector beforeResults;
    std::map<std::string, std::shared_ptr<ngraph::Node>> subgraphNodes;
    std::map<std::string, std::shared_ptr<ngraph::Node>> beforeNodes;
    ngraph::ParameterVector beforeParameters;
    ngraph::ParameterVector subgraphParameters;
    std::set<std::string> out = names;
    for (auto& op : ops) {                                  
        if(out.erase(op->get_friendly_name()) > 0) {  //creates an operation that's clone of op if ops name was in out(copy of names)
            auto subOp = createOp(subgraphNodes, op, subgraphParameters, subgraphResults, beforeNodes, beforeParameters, beforeResults, true);
            for(size_t i = 0; i < op->get_output_size();++i) {
                const auto& set = op->get_output_target_inputs(i); 
                for (auto sel : set) {
                    auto el = sel.get_node();
                    if(out.find(el->get_friendly_name()) == out.end()) {
                        auto res = std::make_shared<ngraph::opset3::Result>(subOp);
                        subgraphResults.push_back(res);

                    }
                }
            }
        }
        if (out.empty()) {
           break;
        }
    }
    if (!out.empty()) {
        for (auto name : out) {
            std::cout << name <<std::endl;
        }
        std::cout << "wasn't found. Stop"<< std::endl;
        return 0;
    }
    std::cout << "All names are valid" << std::endl;
    const auto&  subgraphFunc =  std::make_shared<ngraph::Function>(subgraphResults, subgraphParameters, "Subgraph"); 
    const auto&  beforeFunc =  std::make_shared<ngraph::Function>(beforeResults, beforeParameters, "Beforegraph"); 
    std::cout<<"Subgrap created\n"<<std::endl;
    std::string connection = connectionCheck(subgraphFunc, names); //checks connection
    if(!(connection == "")) {
        std::cout << "Graph isn't complete: " + connection + " dosen't connect to some nodes\n";
        return 0;
    }
    if(file.inputPath.size()) {
       //____calculatin beforegraph______
        ngraph::pass::ConvertPrecision<ngraph::element::Type_t::f16, ngraph::element::Type_t::f32>().run_on_function(beforeFunc);
        beforeFunc->validate_nodes_and_infer_types();
        std::vector<std::vector<std::uint8_t>> beforeInputs;
        beforeInputs.reserve(beforeParameters.size());
        for(auto& param : beforeParameters) {
            beforeInputs.push_back(loadImage(file.inputPath, param));
        }
        std::vector<std::vector<std::uint8_t>> beforeOutputs;
        auto unBeforeOutputs = ngraph::helpers::interpreterFunction(beforeFunc, beforeInputs);
        std::cout<<"Beforegraph calculated\n";
        std::map<std::string, std::string> fileNames;
        size_t last = 0;
        for (auto param : subgraphParameters) {
            const auto& name = param->get_friendly_name();
            auto i = findIndex(beforeResults, param->get_friendly_name());
            if( i != beforeResults.size()) {
                beforeOutputs.push_back(unBeforeOutputs[i]);
            }
            else {
               beforeOutputs.push_back(loadImage(file.inputPath, param));
            }
            auto saveFileName = file.inputFileName;
            for(size_t i =0; i < name.size(); ++i) {
                if(name[i] == '/') {
                    saveFileName += "-";
                }
                else {
                    saveFileName += name[i];
                }
            }
            saveFileName += ".bin";
            fileNames[name] = saveFileName;
            std::ofstream f(saveFileName, std::fstream::out);
            if(!f.is_open()) {
                std::cout << "error while opening save file\n";
                return 1;
            }
            for(auto el : beforeOutputs[last]) {
                f << el;
            }
            ++last;
            f.close();
        }
    //____________________________________
        if(file.calcNet) {
            InferenceEngine::CNNNetwork subNetwork(subgraphFunc);
            InferenceEngine::Core core;
            auto netInputs = subNetwork.getInputsInfo();
            for (auto &input : netInputs) {
                const auto& inputPrecision = input.second->getPrecision();
                if (inputPrecision == InferenceEngine::Precision::FP32 ||
                    inputPrecision == InferenceEngine::Precision::U8) {
                    input.second->setPrecision(InferenceEngine::Precision::FP16);
                }
            }
            auto netOutputs = subNetwork.getOutputsInfo();
            for (auto &output : netOutputs) {
                const auto outputPrecision = output.second->getPrecision();
                if (outputPrecision == InferenceEngine::Precision::FP32) {
                    output.second->setPrecision(InferenceEngine::Precision::FP16);
                }
            }
            auto exeNet = core.LoadNetwork(subNetwork, file.deviceName);
            auto netRequest = exeNet.CreateInferRequest();
            for(auto input : netInputs) {
                auto& name = input.first;
                auto inputBlob = netRequest.GetBlob(name);
                loadBinaryTensorf32(fileNames[name], inputBlob) ;
            }
            netRequest.Infer();
            std::cout << "subgraph calculated\n";
            ngraph::pass::ConvertPrecision<ngraph::element::Type_t::f16, ngraph::element::Type_t::f32>().run_on_function(subgraphFunc);
            subgraphFunc->validate_nodes_and_infer_types();
            auto ref = ngraph::helpers::interpreterFunction(subgraphFunc, beforeOutputs);
            std::cout<<"comparing results: \n";
            for(size_t i =0; i < ref.size();++i) {
                const auto& name = subgraphResults[i]->get_input_node_shared_ptr(0)->get_friendly_name();
                if(compare(ref[i], netRequest.GetBlob(name))) {
                    std::cout<<"\t"<<name<<" fine\n";
                } else {
                    std::cout<<"\t"<<name<<" aren't the same\n";
                }
            }
        }
    }
    return 0;
   
}
