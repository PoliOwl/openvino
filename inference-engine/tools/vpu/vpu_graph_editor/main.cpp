
#include "ngraph/pass/visualize_tree.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "net_pass.h"
#include "ngraph_functions/pass/convert_prc.hpp"
//#include "functional_test_utils/blob_utils.hpp"

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



#define DEBUG 5
#define MAKE_INPUTS 1
#define CALCULATE_SUBGRAPH  1


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

inline float asfloat(uint32_t v) {
    return *reinterpret_cast<float *>(&v);
}

#define EXP_MASK_F32 0x7F800000U
#define EXP_MASK_F16     0x7C00U


//struct for file paths used to get needed data
struct FilePath {

std::string IRpath;
std::string namePath;
std::string weightsPath;
std::string inputPath;
std::string forVisualPath;
std::string inputFileName = "Result";
std::string deviceName =  "MYRIAD";
bool calcNet = false;
//void defult{};

//gets values from command line 
FilePath(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Missing required command line arguments" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (argc == 3) {
       IRpath = argv[1];
       namePath = argv[2];
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
            begin = 3;
        }
        for(int i = begin; i < argc; ++i) {
            std::string operation = argv[i];
            if(operation == "-i") {
                inputPath = argv[++i];
                continue;
            }
            if(operation == "-v") {
                forVisualPath = argv[++i];
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
            std::exit(EXIT_FAILURE);
        }
    }
}

};

class BitMap {
private:
    typedef struct {
        unsigned short type   = 0u;               /* Magic identifier            */
        unsigned int size     = 0u;               /* File size in bytes          */
        unsigned int reserved = 0u;
        unsigned int offset   = 0u;               /* Offset to image data, bytes */
    } BmpHeader;

    typedef struct {
        unsigned int size = 0u;                   /* Header size in bytes      */
        int width = 0, height = 0;                /* Width and height of image */
        unsigned short planes = 0u;               /* Number of colour planes   */
        unsigned short bits = 0u;                 /* Bits per pixel            */
        unsigned int compression = 0u;            /* Compression type          */
        unsigned int imagesize = 0u;              /* Image size in bytes       */
        int xresolution = 0, yresolution = 0;     /* Pixels per meter          */
        unsigned int ncolours = 0u;               /* Number of colours         */
        unsigned int importantcolours = 0u;       /* Important colours         */
    } BmpInfoHeader;

public:
    explicit BitMap(const std::string &filename) {
        BmpHeader header;
        BmpInfoHeader infoHeader;

        std::ifstream input(filename, std::ios::binary);
        if (!input) {
            std::exit(EXIT_FAILURE);
        }

        input.read(reinterpret_cast<char *>(&header.type), 2);

        if (header.type != 'M'*256+'B') {
            std::cerr << "[BMP] file is not bmp type\n";
            std::exit(EXIT_FAILURE);
        }

        input.read(reinterpret_cast<char *>(&header.size), 4);
        input.read(reinterpret_cast<char *>(&header.reserved), 4);
        input.read(reinterpret_cast<char *>(&header.offset), 4);

        input.read(reinterpret_cast<char *>(&infoHeader), sizeof(BmpInfoHeader));

        bool rowsReversed = infoHeader.height < 0;
        _width = infoHeader.width;
        _height = abs(infoHeader.height);

        if (infoHeader.bits != 24) {
            std::cerr << "[BMP] 24bpp only supported. But input has:" << infoHeader.bits << "\n";
            return;
        }

        if (infoHeader.compression != 0) {
            std::cerr << "[BMP] compression not supported\n";
        }

        int padSize = _width & 3;
        char pad[3];
        size_t size = _width * _height * 3;

        _data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());

        input.seekg(header.offset, std::ios::beg);

        // reading by rows in invert vertically
        for (uint32_t i = 0; i < _height; i++) {
            uint32_t storeAt = rowsReversed ? i : (uint32_t)_height - 1 - i;
            input.read(reinterpret_cast<char *>(_data.get()) + _width * 3 * storeAt, _width * 3);
            input.read(pad, padSize);
        }
    }

    ~BitMap() = default;

    size_t _height = 0;
    size_t _width = 0;
    std::shared_ptr<unsigned char> _data;

public:
    size_t size() const { return _width * _height * 3; }
    size_t width() const { return _width; }
    size_t height() const { return _height; }

    std::shared_ptr<unsigned char> getData() {
        return _data;
    }
};

static short f32tof16(float x) {
    static float min16 = asfloat((127 - 14) << 23);

    static float max16 = asfloat(((127 + 15) << 23) | 0x007FE000);
    static uint32_t max16f16 = ((15 + 15) << 10) | 0x3FF;

    union {
        float f;
        uint32_t u;
    } v{};
    v.f = x;

    uint32_t s = (v.u >> 16) & 0x8000;

    v.u &= 0x7FFFFFFF;

    if ((v.u & EXP_MASK_F32) == EXP_MASK_F32) {
        if (v.u & 0x007FFFFF) {
            return s | (v.u >> (23 - 10)) | 0x0200;
        } else {
            return s | (v.u >> (23 - 10));
        }
    }

    float halfULP = asfloat(v.u & EXP_MASK_F32) * asfloat((127 - 11) << 23);
    v.f += halfULP;

    if (v.f < min16 * 0.5F) {
        return s;
    }

    if (v.f < min16) {
        return s | (1 << 10);
    }

    if (v.f >= max16) {
        return max16f16 | s;
    }

    v.u -= ((127 - 15) << 23);

    v.u >>= (23 - 10);

    return v.u | s;
}

std::vector<std::uint8_t> loadImage(const std::string &imageFilename,std::shared_ptr<ngraph::opset3::Parameter>& param) {
    auto& tens = param->output(0).get_tensor();
    //auto layout = tens.get_tensor_layout();

    BitMap reader(imageFilename);

    const auto dims = tens.get_shape();

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

#if DEBUG == 4
void printInput( std::shared_ptr<ngraph::Node> node, int level) {
    for(int i = 0; i < level; ++i) {
        std::cout<<"\t";
    }
    std::cout<<node->get_friendly_name()<<"\t"<<node.get()<<"\n\n";
    for(auto input : node->input_values()) {
        auto inp = input.get_node_shared_ptr();
        printInput(inp, ++level);
    }
}
#endif

bool loadBinaryTensor(const std::string &binaryFilename, InferenceEngine::Blob::Ptr &blob) {
    InferenceEngine::TensorDesc tensDesc = blob->getTensorDesc();
    if (tensDesc.getPrecision() != InferenceEngine::Precision::FP16) {
        std::cout << "loadBinaryTensor error: Input must have FP16 precision" << std::endl;
        return false;
    }

    std::ifstream binaryFile(binaryFilename, std::ios_base::binary | std::ios_base::ate);

    if (!binaryFile) {
        std::cout << "loadBinaryTensor error: While opening a file an error is encountered" << std::endl;
        return false;
    }

    int fileSize = binaryFile.tellg();
    binaryFile.seekg(0, std::ios_base::beg);
    size_t count = blob->size();
    if (fileSize != count * sizeof(float)) {
        std::cout << "loadBinaryTensor error: File contains insufficient items" << std::endl;
        return false;
    }

    if (binaryFile.good()) {
        int16_t *blobDataPtr = std::dynamic_pointer_cast<InferenceEngine::TBlob<int16_t>>(blob)->data();
        for (size_t i = 0; i < count; i++) {
            float tmp = 0.f;
            binaryFile.read(reinterpret_cast<char *>(&tmp), sizeof(float));
            blobDataPtr[i] = f32tof16(tmp);
        }
    } else {
        std::cout << "loadBinaryTensor error: While reading a file an error is encountered" << std::endl;
        return false;
    }
    return true;
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

std::string connectionCheck(const std::shared_ptr<ngraph::Function>& func,const  std::set<std::string>& originalNames) {
    auto&& ops = func->get_ordered_ops();
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

std::shared_ptr<ngraph::Node> make_before_op(std::map<std::string, std::shared_ptr<ngraph::Node>>& before, std::shared_ptr<ngraph::Node>& op, ngraph::ParameterVector& parameters) {
    ngraph::OutputVector args;
    for(auto& input : op->input_values()) {
        auto&& el = input.get_node_shared_ptr();
        auto befNode = before.find(el->get_friendly_name());
        if(befNode != before.end()) {
            args.push_back(befNode->second);
            continue;
        }
        auto&& existingParameter = find(parameters, el->get_friendly_name());
        if(existingParameter.get() != nullptr) {
            args.push_back(existingParameter);
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
    auto newOp = op->clone_with_new_inputs(args);
    newOp->set_friendly_name(op->get_friendly_name());
    before[op->get_friendly_name()] = newOp;
    return newOp;
}

std::shared_ptr<ngraph::Node> createOp(std::map<std::string, std::shared_ptr<ngraph::Node>>& nodes, std::shared_ptr<ngraph::Node>& op, 
                                        ngraph::ParameterVector& parameters,ngraph::ResultVector& results, std::map<std::string, std::shared_ptr<ngraph::Node>>& before, 
                                        ngraph::ParameterVector& beforeParameters, ngraph::ResultVector& beforeResults, bool addNodeToBefore = false){
     std::shared_ptr<ngraph::Node> newOp(nullptr);
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
                    auto&& existingParameter = find(parameters, name);
                    if(existingParameter.get() != nullptr) {
                        args.push_back(existingParameter);
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
                    if(addNodeToBefore && !(el->is_parameter())) {
                        auto before_node = make_before_op(before, el, beforeParameters);
                        beforeResults.push_back(std::make_shared<ngraph::opset3::Result>(before_node));
                    }
                }
                if(op->is_output()) {
                    auto newRes = std::make_shared<ngraph::opset3::Result>(args[0].get_node_shared_ptr());
                    newRes->set_friendly_name(op->get_friendly_name());
                    results.push_back(newRes);
                } else {
                    newOp = op->clone_with_new_inputs(args);
                    newOp->set_friendly_name(op->get_friendly_name());
                    if(!newOp->is_parameter())
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



int main(int argc, char* argv[]) {
    FilePath file(argc, argv);
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
    ngraph::ResultVector subgraphResults;
    ngraph::ResultVector beforeResults;
    std::map<std::string, std::shared_ptr<ngraph::Node>> subgraphNodes;
    std::map<std::string, std::shared_ptr<ngraph::Node>> beforeNodes;
    ngraph::ParameterVector beforeParameters;
    ngraph::ParameterVector subgraphParameters;
    std::set<std::string> out = names;
    bool subgraphNotStarted = true;
    for (auto op : ops) {                                  
        if(out.erase(op->get_friendly_name()) > 0) {  //creates an operation that's clone of op if ops name was in out(copy of names)
            subgraphNotStarted = false;
            auto subOp = createOp(subgraphNodes, op, subgraphParameters, subgraphResults, beforeNodes, beforeParameters, beforeResults, true);
            for(size_t i = 0; i < op->get_output_size();++i) {
                auto set = op->get_output_target_inputs(i); 
                for (auto sel : set) {
                    auto&& el = sel.get_node();
                    if(out.find(el->get_friendly_name()) == out.end()) {
                        auto res = std::make_shared<ngraph::opset3::Result>(subOp);
                        subgraphResults.push_back(res);

                    }
                }
            }
        }
        if(subgraphNotStarted) {
            createOp(beforeNodes, op, beforeParameters,beforeResults, beforeNodes, beforeParameters, beforeResults);
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
    std::string connection = connectionCheck(subgraphFunc, names); //checks connection
    if(!(connection == "")) {
        std::cout << "Graph isn't complete: " + connection + " dosen't connect to some nodes\n";
        return 0;
    }
    if(file.forVisualPath.size()) {
        std::vector<std::shared_ptr<ngraph::Function> > g{subgraphFunc};
        ngraph::pass::VisualizeTree(file.forVisualPath+ "_subgraph.svg").run_on_module(g);
        std::vector<std::shared_ptr<ngraph::Function> > g2{beforeFunc};
        ngraph::pass::VisualizeTree(file.forVisualPath+ "_beforegraph.svg").run_on_module(g2);
    }

    #if MAKE_INPUTS == 1
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
            auto&& name = param->get_friendly_name();
            auto i = findIndex(beforeResults, param->get_friendly_name());
            if( i != beforeResults.size()) {
                beforeOutputs.push_back(unBeforeOutputs[i]);
            }
            else {
               beforeOutputs.push_back(loadImage(file.inputPath, param));
            }
            fileNames[name] = file.inputFileName+name+".bin";
            std::ofstream f(fileNames[name]);
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
                const auto inputPrecision = input.second->getPrecision();
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
                if(!loadBinaryTensor(fileNames[name], inputBlob)) {
                    return 1;
                }
            }
            netRequest.Infer();
            std::cout << "subgraph calculated\n";
        }

    
    }
    #endif
    return 0;
   
}
