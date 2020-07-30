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
#include "ngraph/pass/visualize_tree.hpp"
#include "cnn_network_ngraph_impl.hpp"


#define DEBUG 0


void readNames(std::set<std::string>&, const std::string&); //read subgraph nodes names from file
bool connection_check(std::shared_ptr<ngraph::Function>&, const  std::set<std::string>& ); //checks if graph is complete. If it is, return "", if not - name of node that is unnable to rich from every node
void visit(const std::shared_ptr<ngraph::Node>&, std::map<std::string, bool>&); 

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

template <typename T>
std::shared_ptr<T> find(std::vector<std::shared_ptr<T>>& vector,const std::string& node) { 
    for(auto el : vector) {
        if(el->get_friendly_name() == node) {
            return el;
        }
    }
    return std::shared_ptr<T>(nullptr);
}

#if DEBUG == 4
void print_input( std::shared_ptr<ngraph::Node> node, int level) {
    for(int i = 0; i < level; ++i) {
        std::cout<<"\t";
    }
    std::cout<<node->get_friendly_name()<<"\t"<<node.get()<<"\n\n";
    for(auto input : node->input_values()) {
        auto inp = input.get_node_shared_ptr();
        print_input(inp, ++level);
    }
}
#endif

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

void visit(const std::shared_ptr<ngraph::Node>& node, std::map<std::string, bool>& map) {
    map[node->get_friendly_name()] = true;
    for(auto& inp : node->input_values()) {
        auto&& input = inp.get_node_shared_ptr();
        if(map[input->get_friendly_name()] == false) {
            visit(input, map);
        }
    }
    if(node->is_output() == false) {
        for(size_t i = 0; i < node->get_output_size();++i) {
            auto&& set = node->get_output_target_inputs(i); 
            for (auto& sel : set) {
                auto el = sel.get_node();
                if(map[el->get_friendly_name()] == false) {
                    auto&& elPtr = el->output(0).get_node_shared_ptr();
                    ///auto ptr = std::make_shared<ngraph::Node>(*el);
                    visit(elPtr, map);
                }
            }
        }
    }
    
}

std::string connection_check(const std::shared_ptr<ngraph::Function>& func,const  std::set<std::string>& original_names) {
    auto&& ops = func->get_ordered_ops();
    std::map<std::string, bool> visited;
    for(auto op : ops) {
        visited[op->get_friendly_name()] = false;
    }
    visit(ops[0], visited);
    for(auto el : visited) {
        if(el.second == false) {
            if(original_names.count(el.first)) {
                return el->first;
            }
        }
    }
    return "";
}




int main(int argc, char* argv[]) {
    filePaths file(argc, argv);
    std::set<std::string> names; //set of subgraphs nodes names
    readNames(names, file.namePath);
    #if DEBUG == 2
    for(auto name: names) {
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
    std::set<std::string> out = names;
    for (auto op : ops) {                                  
        if(out.erase(op->get_friendly_name()) > 0) {  //creates an operation that's clone of op if ops name was in out(copy of names)
            std::shared_ptr<ngraph::Node> new_op;
            if(!(op->is_parameter())) {
                ngraph::OutputVector args;
                for(auto& element : op->input_values()) {
                    auto&& el = element.get_node_shared_ptr(); 
                    auto&& name = el->get_friendly_name();
                    auto subNode = nodes.find(name);
                    if(subNode != nodes.end()) {
                        args.push_back(subNode->second);
                        continue;
                    }
                    auto&& existing_parameter = find(parameters, name);
                    if(existing_parameter.get() != nullptr) {
                        args.push_back(existing_parameter);
                        continue;
                    }
                    if(el->is_constant()) {
                        const auto&& constanta= dynamic_cast<ngraph::opset3::Constant*>(el.get());
                        const auto& subConstant = std::make_shared<ngraph::opset3::Constant>(constanta->get_element_type(), constanta->get_shape(), constanta->get_value_strings());
                        subConstant->set_friendly_name(el->get_friendly_name());
                        nodes[el->get_friendly_name()] = subConstant;
                        args.push_back(subConstant);
                        continue;
                    }
                    const auto& stubParameter = std::make_shared<ngraph::opset3::Parameter>(element.get_element_type(), element.get_partial_shape());
                    stubParameter->set_friendly_name(name);
                    parameters.push_back(stubParameter);
                    args.push_back(stubParameter);
                }
                new_op = op->clone_with_new_inputs(args);
                new_op->set_friendly_name(op->get_friendly_name());
                nodes[op->get_friendly_name()] = new_op;
            }
            else {
                const auto& stubParameter = std::make_shared<ngraph::opset3::Parameter>(op->get_element_type(), op->get_shape());
                stubParameter->set_friendly_name(op->get_friendly_name());
                new_op = stubParameter;
                parameters.push_back(stubParameter);
            }
            for(size_t i = 0; i < op->get_output_size();++i) {
                auto set = op->get_output_target_inputs(i); 
                for (auto sel : set) {
                    auto&& el = sel.get_node();
                    if(out.find(el->get_friendly_name()) == out.end()) {
                        auto res = std::make_shared<ngraph::opset3::Result>(new_op);
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
    std::cout << "All names are valid" << std::endl;
    const auto&  subgraphFunc =  std::make_shared<ngraph::Function>(results, parameters, "Subgraph"); 
    std::vector<std::shared_ptr<ngraph::Function>> fv{subgraphFunc};
    ::ngraph::pass::VisualizeTree("home/googlenetv4/faster.svg").run_on_module(fv);
    std::cout<<"visualization created\n";
    std::cout<<"Subgrap created\n"<<std::endl;
    #if DEBUG == 3
    auto s = subgraphFunc->get_ordered_ops();
    for(auto op : s) {
        std::cout<<op->get_friendly_name()<<'\n';
    }
    #endif
    std::string connection = connection_check(subgraphFunc, names); //checks connection
    if(!(connection == "")) {
        std::cout << "Graph isn't complete: " + connection + " dosen't connect to some nodes\n";
        return 0;
    }

    return 0;
}
