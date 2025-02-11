#include <stdio.h>
#include "opencv_dnn_wrapper.h"

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

    // Create input array with the provided data (size should match the model's expected input)
    float input_data[1][220] = {
        {-4.7023e+02, -3.9791e+02, -2.6867e+02, -2.2329e+02, -2.2694e+02,
        -2.5677e+02, -2.8345e+02, -3.1838e+02, -3.8183e+02, -4.1898e+02,
        -4.3908e+02,  2.3684e+01,  1.2044e+00, -1.3274e+01,  3.2263e+00,
         5.2652e+01,  6.8911e+01,  5.2675e+01,  4.2756e+01,  4.2379e+01,
         4.0567e+01,  3.7221e+01,  4.6790e+01,  5.6301e+01,  5.1584e+01,
         5.8641e+01,  4.9718e+01,  4.4658e+01,  3.0980e+01,  4.3094e+01,
         4.5441e+01,  4.9720e+01,  4.8439e+01,  3.2895e+01,  2.3849e+01,
         2.1433e+01,  3.0220e+01, -4.7311e+00, -4.3485e+01, -2.0728e+01,
         1.4020e+01,  3.4952e+01,  3.4697e+01,  3.2183e+01,  1.1721e+01,
        -5.1574e+00, -1.7550e+01, -1.3550e+01, -1.6937e+01, -2.3524e+00,
         9.2503e+00,  4.7933e+00,  7.5747e+00,  6.2456e+00,  6.9504e+00,
         2.2026e+01,  7.5991e+00, -8.2416e+00, -1.0130e+01, -6.7824e+00,
        -7.2398e-01, -9.2939e+00, -1.6500e+01,  4.6027e+00,  1.7310e+01,
         2.0090e+01,  8.3415e+00,  1.0659e+01, -4.6047e+00, -1.0311e+01,
        -9.0712e-01,  1.8519e+00,  1.4625e+00, -6.0048e+00, -4.7871e+00,
         7.4717e+00,  9.5000e+00,  1.0746e+01,  2.0678e+01,  1.7160e+01,
         8.7322e+00,  1.0434e+00, -1.3466e+01, -7.6598e+00,  4.7841e+00,
         9.1676e+00,  8.2275e+00,  1.3158e+01,  6.1295e+00,  2.2968e+01,
         2.9641e+01,  2.2531e+01,  8.0063e+00, -6.0614e+00, -1.1322e+01,
        -5.1759e+00,  9.2746e-01,  5.2269e+00,  8.2493e+00,  1.6834e+01,
         2.2845e+01,  3.7126e+01,  4.0416e+01,  2.4140e+01,  1.1368e+01,
        -3.2864e+00,  3.4956e-01,  3.4749e+00,  9.6463e+00,  1.6353e+01,
         1.1140e+01,  1.0811e+01,  2.9823e+01,  3.5326e+01,  1.9376e+01,
         1.2849e+01,  8.8262e+00,  4.2399e+00,  3.4447e+00,  1.3047e+00,
         3.9029e+00,  6.1525e+00,  6.6260e+00,  1.8886e+01,  2.5472e+01,
         2.3489e+01,  6.0757e+00, -1.7167e+00,  2.9856e+00,  1.0371e+01,
        -1.8084e-02, -1.0189e+00,  8.2113e+00, -5.3026e+00, -1.6532e+01,
        -1.9610e+01, -2.7385e+00,  1.3773e+00, -1.0547e+01,  6.3796e-01,
         8.3483e+00,  2.8930e+00,  2.9654e+00,  2.6900e+00, -1.6715e+00,
        -1.5725e+01, -2.0332e+01,  4.5640e-01,  1.9762e+01,  7.8170e+00,
         6.8455e+00,  5.4484e+00,  9.3774e-01,  4.2341e+00,  3.1754e+00,
         6.3440e-01,  2.4271e+00,  3.9913e+00,  1.8270e+01,  2.7428e+01,
         2.2247e+01,  2.3493e+01,  1.4955e+01, -5.9615e-02,  1.8791e+00,
         1.8365e+00,  9.1329e-01,  7.5386e+00,  1.2098e+01,  2.0649e+01,
         2.8640e+01,  2.7829e+01,  1.7634e+01,  4.5398e+00, -1.1725e+00,
        -3.0654e+00, -1.6276e+00, -8.8405e+00, -1.4909e+01, -1.6810e+01,
        -9.6745e+00,  1.2219e+01,  1.9337e+01,  7.9813e+00, -4.6666e+00,
        -5.4717e+00, -6.6991e+00,  4.1536e+00,  1.2574e+00, -6.7894e+00,
        -8.6784e+00, -5.6603e+00,  6.8251e+00,  2.2397e+01,  1.5797e+01,
        -1.7831e+00, -1.0462e+00, -4.0551e+00,  3.6924e+00,  2.7535e+00,
         9.3778e+00,  1.0841e+01,  5.5958e+00,  5.4953e+00,  1.2759e+01,
         6.7545e+00, -2.0714e+00,  4.1324e+00,  1.4961e+00,  1.4004e+00,
         2.4994e+00,  2.0502e+01,  2.2909e+01,  1.3939e+01,  1.5495e+00,
        -4.7329e+00, -7.6568e+00, -4.5644e+00,  5.4969e+00,  2.1390e-01}
    };

    printf("Input data loaded successfully!\n");

    // Create input blob from data
    printf("Creating input blob...\n");
    Mat* input_blob = create_input_blob((float*)input_data, 1, 220, 1);  // Corrected shape for input
    if (input_blob == NULL) {
        printf("Failed to create input blob!\n");
        release_net(net);
        return 1;
    }
    printf("Input blob created successfully!\n");

    // Run inference
    printf("Running inference...\n");
    Mat* output = dnn_forward(net, input_blob);
    if (output == NULL) {
        printf("DNN inference failed!\n");
    } else {
        // Assuming get_max_class is correctly defined in the wrapper
        int predicted_class = get_max_class(output);
        printf("Predicted class: %d\n", predicted_class);
        release_mat(output);  // Don't forget to free the output after use
    }

    // Cleanup
    printf("Cleaning up...\n");
    release_mat(input_blob);  // Release input blob
    release_net(net);         // Release the loaded model

    printf("Program finished successfully!\n");
    return 0;
}
