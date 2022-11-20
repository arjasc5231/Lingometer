#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_

// spectrogram 생성을 위한 상수
constexpr int kMaxAudioSampleSize = 512;  // 오디오 최대 길이(2의 제곱수)
constexpr int kAudioSampleFrequency = 16000;

// spectogram generator에서 사용하는 변수들이라 이름을 맘대로 바꾸기 쉽지 않음
constexpr int kFeatureSliceSize = 40;
constexpr int kFeatureSliceCount = 49;
constexpr int kFeatureElementCount = (kFeatureSliceSize * kFeatureSliceCount);
constexpr int kFeatureSliceStrideMs = 20;
constexpr int kFeatureSliceDurationMs = 30;

constexpr int spec_dim = 91*40;  // SV용 spec이랑 WC용 spec이 달라서 구분할 필요가 있음
constexpr int dvec_dim = 50;

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_
