#include "feature_provider.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "spectrogram_generator.h"
#include "speaker_verification_model_settings.h"


// 객체 인자로 스펙트로그램 크기와 스펙트로그램이 저장될 포인터를 받는다.
FeatureProvider::FeatureProvider(int feature_size, int8_t* feature_data)
    : feature_size_(feature_size),
      feature_data_(feature_data),
      is_first_run_(true) {
  // Initialize the feature data to default values.
  for (int n = 0; n < feature_size_; ++n) { feature_data_[n] = 0; }
}

FeatureProvider::~FeatureProvider() {}

// 오디오 입력을 받아와 필요한 부분의 스펙트로그램을 생성하는 함수
int FeatureProvider::PopulateFeatureData(tflite::ErrorReporter* error_reporter, int16_t* audio, int audio_len, int8_t* spectrogram) {
  if (is_first_run_) {
    TfLiteStatus init_status = InitializeMicroFeatures(error_reporter);
    if (init_status != kTfLiteOk) { return init_status; }
    is_first_run_ = false;
  }

  int total_frame = (((audio_len*1000 / kAudioSampleFrequency)-kFeatureSliceDurationMs) / kFeatureSliceStrideMs)+1; //총 window 개수
  
  for (int frame = 0; frame < total_frame; ++frame) {
    const int32_t slice_start_ms = (frame * kFeatureSliceStrideMs);

    // 해당 프레임(윈도우)의 음성 옮기기
    int16_t audio_samples[kMaxAudioSampleSize];
    int audio_samples_size = kMaxAudioSampleSize;
    int duration = kFeatureSliceDurationMs * (kAudioSampleFrequency / 1000);
    int start_sample = slice_start_ms * (kAudioSampleFrequency / 1000);
    for (int i = 0; i < duration; ++i) { audio_samples[i] = audio[start_sample + i]; } // pointer indexing으로 하면 안되나?

    // 스펙트로그램 생성
    int8_t* new_frame_data = spectrogram + (frame * kFeatureSliceSize);
    TfLiteStatus generate_status = GenerateMicroFeatures(error_reporter, audio_samples, audio_samples_size, kFeatureSliceSize, new_frame_data, NULL);
    if (generate_status != kTfLiteOk) {return generate_status;}
  }
  return total_frame*kFeatureSliceSize;
}
