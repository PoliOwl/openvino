#include <iostream>
#include <inference_engine.hpp>
#include <generic_ie.hpp>
#include "net_pass.h"
#include <fstream>
#include <set>
#include <vector>
#include <string>
#include <cstdlib>
#include <map>
#include <ngraph/opsets/opset3.hpp>
#include "cnn_network_ngraph_impl.hpp"


#define DEBUG 3


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
           NamesCont.insert(std::move(s));
       }
    }
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



int main(int argc, char* argv[]) {
    filePaths file(argc, argv);
    std::set<std::string> out;
    readNames(out, file.namePath);
    #if DEBUG == 2
    for(auto name: input) {
        std::cout << name<<'\n';
    }
    std::cout << "___________________________";
    #endif
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
    ngraph::ResultVector results;
    std::map<std::string, std::shared_ptr<ngraph::Node>> nodes;
    ngraph::ParameterVector parameters;
    for (auto op : ops) {
        if(out.erase(op->get_friendly_name()) > 0) {
            if(!(op->is_parameter())) {
                ngraph::OutputVector args;
                for(auto& element : op->input_values()) {
                    auto&& el = element.get_node_shared_ptr(); 
                    auto&& name = el->get_friendly_name();
                    auto subNode = nodes.find(name);
                    if(subNode != nodes.end()) {
                        args.push_back(subNode->second->output(element.get_index()));
                       // std::cout<<subNode->second->output(element.get_index())<<'\n'<<subNode->second<<'\n';
                        continue;
                    }
                    auto&& existing_parameter = find(parameters, name);
                    if(existing_parameter.get() != nullptr) {
                        //std::cout<<"Here: "<<name<<'\t'<<op->get_friendly_name()<<'\n';
                        args.push_back(existing_parameter->output(element.get_index()));
                        continue;
                    }
                    const auto& stubParameter = std::make_shared<ngraph::opset3::Parameter>(element.get_element_type(), element.get_partial_shape());
                    stubParameter->set_friendly_name(name);
                    parameters.push_back(stubParameter);
                    args.push_back(stubParameter);
                }
                auto&& new_op = op->copy_with_new_inputs(args);
                new_op->set_friendly_name(op->get_friendly_name());
                nodes[op->get_friendly_name()] = new_op;
            }
            else {
                const auto& stubParameter = std::make_shared<ngraph::opset3::Parameter>(op->get_element_type(), op->get_shape());
                stubParameter->set_friendly_name(op->get_friendly_name());
                parameters.push_back(stubParameter);
            }
            for(size_t i = 0; i < op->get_output_size();++i) {
                auto set = op->get_output_target_inputs(i); 
                for (auto sel : set) {
                    auto&& el = sel.get_node();
                    if(out.find(el->get_friendly_name()) == out.end()) {
                        auto res = std::make_shared<ngraph::opset3::Result>(op);
                        res->set_friendly_name(el->get_friendly_name());
                        results.push_back(res);

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
    #if DEBUG == 3
    std::cout<<"\ninput:\n";
    for(auto el : nodes) {
        std::cout << "\t" << el.second->get_friendly_name() << "\t" <<el.second.get()<<'\n';
        for(auto par_c : el.second->input_values()) {
            auto par = par_c.as_single_output_node(); 
            std::cout<<"\t\t"<<par->get_friendly_name()<<"\t"<<par.get()<<"\n\n";
        }
    }
    std::cout<<"\nparameters:\n";
    for(auto par : parameters) {
        std::cout<<"\t"<<par->get_friendly_name()<<'\t'<<par.get()<<"\n";
        for(auto par_c : par->input_values()) {
            auto par_n = par_c.as_single_output_node(); 
            std::cout<<"\t\t"<<par_n->get_friendly_name()<<"\t"<<par_n.get()<<"\n\n";
        }
    }
    std::cout<<"\nresults:\n";
    for(auto res : results){
        std::cout<<"\t"<<res->get_friendly_name()<<'\t'<<res.get()<<"\n";
        for(auto par_c : res->input_values()) {
            auto par = par_c.as_single_output_node(); 
            std::cout<<"\t\t"<<par->get_friendly_name()<<"\t"<<par.get()<<"\n\n";
        }
    }
    #endif
    std::cout << "All names are valid" << std::endl;
    ngraph::Function subgraphFunc(results, parameters, "Subgraph");
    std::cout<<"Subgrap created"<<std::endl;
    #if DEBUG == 3
    auto s = subgraphFunc.get_ordered_ops();
    for(auto op : s) {
        std::cout<<op->get_friendly_name()<<'\n';
    }
    #endif
    return 0;
}
