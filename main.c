
#include <stdio.h>
#include "opencv_dnn_wrapper.h"
#include "input_data.h"

#define NUM_SETS 5  // Define the number of input data sets
#define DATA_SIZE 220  // Define the size of each data set

int main() {
    const char* model_path = "audio_classifier.onnx";
    printf("Loading model...\n");

    // Load ONNX model
    Net* net = load_dnn_model(model_path, "");
    if (net == NULL) {
        printf("Failed to load DNN model!\n");
        return 1;
    }
    printf("Model loaded successfully!\n");
    printf("C Code \n");

    float input_data[NUM_SETS][DATA_SIZE];
    get_input_data(input_data);

    for (int set = 0; set < NUM_SETS; set++) {
        //printf("Processing Input Data Set %d...\n", set + 1);

        Mat* input_blob = create_input_blob((float*)input_data[set], 1, DATA_SIZE, 1);
        if (!input_blob) {
            printf("Failed to create input blob for set %d!\n", set);
            continue;
        }

        //printf("Running inference for set %d...\n", set + 1);
        Mat* output = dnn_forward(net, input_blob);
        if (!output) {
            printf("Inference failed for set %d!\n", set);
        } else {
            int predicted_class = get_max_class(output);
            printf("Predicted class for set %d: %d\n", set, predicted_class);
            release_mat(output);
        }

        release_mat(input_blob);
    }

    release_net(net);
    printf("Cleanup complete.\n");

    return 0;
}
