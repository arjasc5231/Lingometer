#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"


class FeatureProvider {
 public:
  FeatureProvider(int feature_size, int8_t* feature_data);
  ~FeatureProvider();

  int PopulateFeatureData(tflite::ErrorReporter* error_reporter, int16_t* audio, int audio_len, int8_t* spectrogram);

 private:
  int feature_size_;
  int8_t* feature_data_;
  bool is_first_run_;
};
