#ifndef OPENCV_DNN_WRAPPER_H
#define OPENCV_DNN_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Net Net;
typedef struct Mat Mat;

Net* load_dnn_model(const char* model_path, const char* config_path);
Mat* dnn_forward(Net* net, Mat* input_blob);
Mat* convert_image_to_blob(Mat* img, double scale, const int* size, const double* mean, int swapRB, int crop);
Mat* load_image(const char* filename);
void release_net(Net* net);
void release_mat(Mat* mat);

// ✅ Add missing function declarations
Mat* create_input_blob(float* data, int batch, int height, int width);
int get_max_class(Mat* output);

#ifdef __cplusplus
}
#endif

#endif // OPENCV_DNN_WRAPPER_H



