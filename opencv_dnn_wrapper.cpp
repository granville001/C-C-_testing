#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <cstdio>

using namespace cv;
using namespace cv::dnn;

extern "C" {

// Load the ONNX model
Net* load_dnn_model(const char* model_path, const char* config_path) {
    return new Net(readNet(model_path, config_path));
}

// Create an input blob from raw float data
Mat* create_input_blob(float* data, int channels, int height, int width) {
    // Create a 2D matrix with dimensions (height x width)
    Mat* blob = new Mat(Size(width, height), CV_32FC(channels), data);

    // Reshape it to [1, channels, height, width] for batch size 1
    // [1, 20, 11] would correspond to [batch_size, height, width]
    // The 1st dimension is the batch size, which is set to 1 here.
    blob = new Mat(blob->reshape(channels, 1));  // Reshaping to match the expected format

    return blob;
}


// Run inference
Mat* dnn_forward(Net* net, Mat* input_blob) {
    // Check if network and input blob are valid
    if (!net || !input_blob) {
        printf("Failed to load model or input blob is null\n");
        return nullptr;
    }

    // Debugging: Print the input blob's shape
    //printf("Input Blob shape: %d x %d\n", input_blob->rows, input_blob->cols);

    // Set the input to the network
    net->setInput(*input_blob, "input");

    // Run forward pass to get the output
    // printf("Running forward pass...\n");
    Mat* output = new Mat(net->forward("output"));
    if (!output) {
        printf("Failed to get output from forward pass\n");
        return nullptr;
    }

    return output;
}

// Get the class with the highest probability
int get_max_class(Mat* output) {
    if (!output) return -1;
    Point class_id;
    minMaxLoc(*output, nullptr, nullptr, nullptr, &class_id);
    return class_id.x;
}

// Release memory
void release_net(Net* net) { delete net; }
void release_mat(Mat* mat) { delete mat; }

}
