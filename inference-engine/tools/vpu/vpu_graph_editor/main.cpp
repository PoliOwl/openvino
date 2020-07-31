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


#define DEBUG 3


void readNames(std::set<std::string>&, const std::string&); //read subgraph nodes names from file
bool connection_check(std::shared_ptr<ngraph::Function>&, const  std::set<std::string>& ); //checks if graph is complete. If it is, return "", if not - name of node that is unnable to rich from every node
void visit(const std::shared_ptr<ngraph::Node>&, std::map<std::string, bool>&); 
std::shared_ptr<ngraph::Node> make_before_op(std::map<std::string, std::shared_ptr<ngraph::Node>>& before, std::shared_ptr<ngraph::Node>& op, ngraph::ParameterVector& parameters); //creates node for beforegraph
std::shared_ptr<ngraph::Node> create_op(std::map<std::string, std::shared_ptr<ngraph::Node>>& nodes, std::shared_ptr<ngraph::Node>& op, 
                                        ngraph::ParameterVector& parameters, std::map<std::string, std::shared_ptr<ngraph::Node>>& before, 
                                        ngraph::ParameterVector& before_parameters, ngraph::ResultVector& before_results, bool add_node_to_before); //creats operation, 
//if add_node_to_before = false, then it doesn't uses before.. parametrs  and only creates node, otherwise invokes make_before_op using before and before before_parameters and adds result to before_results 

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
                return el.first;
            }
        }
    }
    return "";
}

std::shared_ptr<ngraph::Node> make_before_op(std::map<std::string, std::shared_ptr<ngraph::Node>>& before, std::shared_ptr<ngraph::Node>& op, ngraph::ParameterVector& parameters) {
    ngraph::OutputVector args;
    for(auto& input : op->input_values()) {
        auto&& el = input.get_node_shared_ptr();
        auto befNode = before.find(el->get_friendly_name());
        if(befNode != before.end()) {
            args.push_back(befNode->second);
            continue;
        }
        auto&& existing_parameter = find(parameters, el->get_friendly_name());
        if(existing_parameter.get() != nullptr) {
            args.push_back(existing_parameter);
            continue;
        }
        if(el->is_constant()) {
            const auto&& constanta= dynamic_cast<ngraph::opset3::Constant*>(el.get());
            const auto& subConstant = std::make_shared<ngraph::opset3::Constant>(constanta->get_element_type(), constanta->get_shape(), constanta->get_value_strings());
            subConstant->set_friendly_name(el->get_friendly_name());
            before[el->get_friendly_name()] = subConstant;
            args.push_back(subConstant);
            continue;
        }
        if(el->is_parameter()) {
            const auto& stubParameter = std::make_shared<ngraph::opset3::Parameter>(input.get_element_type(), input.get_partial_shape());
            stubParameter->set_friendly_name(el->get_friendly_name());
            parameters.push_back(stubParameter);
            args.push_back(stubParameter);
            continue;
        }
        args.push_back(make_before_op(before, el, parameters));
    }
    auto new_op = op->clone_with_new_inputs(args);
    new_op->set_friendly_name(op->get_friendly_name());
    before[op->get_friendly_name()] = new_op;
    return new_op;
}

std::shared_ptr<ngraph::Node> create_op(std::map<std::string, std::shared_ptr<ngraph::Node>>& nodes, std::shared_ptr<ngraph::Node>& op, 
                                        ngraph::ParameterVector& parameters, std::map<std::string, std::shared_ptr<ngraph::Node>>& before, 
                                        ngraph::ParameterVector& before_parameters, ngraph::ResultVector& before_results, bool add_node_to_before = false){
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
                    if(add_node_to_before && !(el->is_parameter())) {
                        auto before_node = make_before_op(before, el, before_parameters);
                        before_results.push_back(std::make_shared<ngraph::opset3::Result>(before_node));
                    }
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
    return new_op;
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
    ngraph::ResultVector subgraph_results;
    ngraph::ResultVector before_results;
    std::map<std::string, std::shared_ptr<ngraph::Node>> subgraph_nodes;
    std::map<std::string, std::shared_ptr<ngraph::Node>> before_nodes;
    ngraph::ParameterVector before_parameters;
    ngraph::ParameterVector subgraph_parameters;
    std::set<std::string> out = names;
    bool subgraph_not_started = true;
    for (auto op : ops) {                                  
        if(out.erase(op->get_friendly_name()) > 0) {  //creates an operation that's clone of op if ops name was in out(copy of names)
            subgraph_not_started = false;
            auto sub_op = create_op(subgraph_nodes, op, subgraph_parameters, before_nodes, before_parameters, before_results, true);
            for(size_t i = 0; i < op->get_output_size();++i) {
                auto set = op->get_output_target_inputs(i); 
                for (auto sel : set) {
                    auto&& el = sel.get_node();
                    if(out.find(el->get_friendly_name()) == out.end()) {
                        auto res = std::make_shared<ngraph::opset3::Result>(sub_op);
                        subgraph_results.push_back(res);

                    }
                }
            }
        }
        if(subgraph_not_started) {
            create_op(before_nodes, op, before_parameters, before_nodes, before_parameters, before_results);
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
    const auto&  subgraphFunc =  std::make_shared<ngraph::Function>(subgraph_results, subgraph_parameters, "Subgraph"); 
    const auto&  beforeFunc =  std::make_shared<ngraph::Function>(before_results, before_parameters, "Beforegraph"); 
    std::cout<<"Subgrap created\n"<<std::endl;
    #if DEBUG == 3
    std::cout<<"\n______subgraph______________\n";
    auto s = subgraphFunc->get_ordered_ops();
    for(auto op : s) {
        std::cout<<op->get_friendly_name()<<'\n';
    }
    std::cout<<"\n______Beforegraph______________\n";
    auto bops = beforeFunc->get_ordered_ops();
    for(auto op : bops) {
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
