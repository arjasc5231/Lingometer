#include <PDM.h> //마이크 입력 관련 라이브러리
#include "protothreads.h" //PT 라이브러리
#include <SPI.h> //SD카드 라이브러리
#include <SD.h> //SD카드 라이브러리

#include "Const.h" //기본 상수값 필요한 게 생기면 여기에.

#include "OLED.h" //OLED 출력 조작 관련 함수

#include "CountingWords.h" //단어 세는 함수
#include "LearningVoice.h" //목소리 학습 함수


#include <TensorFlowLite.h> //이 아래로 WC, SV 관련 모델
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "micro_features_micro_features_generator.h" //이거랑 아래거는 스펙토그램용
#include "tensorflow/lite/micro/micro_error_reporter.h" 


#include "speaker_verification_model_settings.h"
#include "speaker_verification_model.h"
#include "word_counting_model.h"


volatile int mode=1;
// 1: 측정중 0: 측정 중지 2: 학습
volatile int light=1;
// 1: 화면 켜짐  0: 화면 꺼짐

volatile int button1_chk=1; //버튼1 조작용 변수, 눌리면 0, 아니면 1
volatile int button2_chk=1; //버튼2 조작용 변수, 눌리면 0, 아니면 1
volatile unsigned long b1_in_time=1;
volatile unsigned long b1_out_time=1;
volatile unsigned long b2_in_time=1;
volatile unsigned long b2_out_time=1;
volatile unsigned long last_control=1;
volatile unsigned long now=1;

volatile unsigned int num_words=100; // 측정된 단어 수
short Buffer[256]; // 음성 신호 입력받을 변수
short Buffer2[512]; //음성 신호 임시 저장용 변수
volatile int w=0; //Buffer2 관리용
volatile int Read; //음성 신호 입력용 변수
volatile int conv2spect=0; //스펙토그램 변환 확인용 변수
File spectogramFile; //이 파일에 스펙토그램 넣을 예정임.

volatile int startSVWC=0; //단어수 카운트 시작 여부 알려줌.
volatile int w2=0; //단어수 카운트 관리용
volatile int w3=0;

const int chipSelect = 10; //SD카드 핀번호 알려줌.

volatile int tmp_count=0;
volatile int chk_counted=0;

const int g_yes_feature_data_slice_size = 40; //만들 스펙토그램 사이즈 91*40?
int8_t yes_calculated_data[g_yes_feature_data_slice_size]; //만든 스펙토그램 저장 공간
 const int g_yes_30ms_sample_data_size = 480; //인풋 오디오 데이터 사이즈, 29280?
/////////이 아래로 SVWC 전역변수//////////
// 에러 리포터 전역변수 선언
tflite::ErrorReporter* error_reporter = nullptr;

float total_words=0; // 총 단어 수
float enroll_dvec[dvec_dim]; // 화자 등록 d-vector (normalized)
float thres = 0; // SV 역치

// 스펙트로그램
constexpr int spec_len = 91*40;
int spec[spec_len]; // 실전에서는 (n*91)&40이 되도록 제로패딩

// SV 모델 전역변수 선언
const tflite::Model* SV_model = nullptr;
tflite::MicroInterpreter* SV_interpreter = nullptr;
TfLiteTensor* SV_model_input = nullptr;
constexpr int SV_tensorArenaSize = 22500; // 모델에 따라 크기 조정. 나중에 상수파일로 옮기기
uint8_t SV_tensor_arena[SV_tensorArenaSize];
int8_t* SV_model_input_buffer = nullptr;
float score1=0;
float score2=0;

// WC 모델 전역변수 선언
const tflite::Model* WC_model = nullptr;
tflite::MicroInterpreter* WC_interpreter = nullptr;
TfLiteTensor* WC_model_input = nullptr;
constexpr int WC_tensorArenaSize = 70000; // 모델에 따라 크기 조정. 나중에 상수파일로 옮기기
uint8_t WC_tensor_arena[WC_tensorArenaSize];
//int8_t WC_feature_buffer[kFeatureElementCount];
int8_t* WC_model_input_buffer = nullptr;

// 모델의 실행 시간을 체크할 전역변수
unsigned long end_time;
unsigned long start_time;

// 아두이노 내장 LED 설정
const int RED = 22;
const int GREEN = 23;
const int BLUE = 24;

// 모든 Op load. 필요한 Op만 로드해서 메모리를 줄일수도 있음
tflite::AllOpsResolver resolver;
//////////////여기까지 SVWC전역변수////////////


pt ptState;
int stateThread(struct pt* pt){
  PT_BEGIN(pt);
  for(;;){
    button1_chk=digitalRead(2);
    button2_chk=digitalRead(4);
    PT_YIELD(pt);
    }
    PT_END(pt);
  }

pt ptButton1Time; // 버튼1 프레스 시간 확인 및 조작용 PT
int button1TimeThread(struct pt* pt){
  PT_BEGIN(pt);
  for(;;){
      PT_YIELD(pt);
      
      PT_WAIT_UNTIL(pt, button1_chk==0);
      b1_in_time=millis();
      last_control=millis();
      PT_YIELD(pt);
      
      PT_WAIT_UNTIL(pt,button1_chk==1);
      b1_out_time=millis();
      last_control=millis();
      light=1;
      
      if(b1_out_time-b1_in_time<2000){
        if(mode==0){mode=1;}
        else if(mode=1){mode=0;} //짧게 누르면 측정 일시정지/ 재개하고
        } else {num_words=0; SD.remove("Specto.txt");} //길게 누르면 측정 초기화한다.
      PT_YIELD(pt);
    }
  PT_END(pt);
  }


pt ptButton2Time; // 버튼2 프레스 시간 확인 및 조작용 PT
int button2TimeThread(struct pt* pt){
  PT_BEGIN(pt);

  for(;;){
      PT_YIELD(pt);
      
      PT_WAIT_UNTIL(pt, button2_chk==0);
      b2_in_time=millis();
      last_control=millis();
      PT_YIELD(pt);
      Serial.println("button2ininininin");
      
      PT_WAIT_UNTIL(pt,button2_chk==1);
      b2_out_time=millis();
      last_control=millis();
      Serial.println("button2outoutoutoutoutout");
      
      if(b2_out_time-b2_in_time<2000){
        if(light==0){
        light=1;
        startSVWC=1; //불이 켜지면 SVWC 시작
        }
        else if(light==1){light=0;}
         //짧게 누르면 불이 켜지거나 꺼지고
        else{
          mode=2;
          light=1;;
          } //길게 누르면 불이 무조건 켜지면서 모드 2
      PT_YIELD(pt);     
      
    }
  }
  PT_END(pt);
}

//마이크 입력 저장
pt ptRecording;
int recordingThread(struct pt* pt){
  PT_BEGIN(pt);
  for(;;){
    if(mode==1 && startSVWC==0){
      if(Read){
        for(int i=0; i<Read; i++){
          Buffer2[w]=Buffer[i];
          w++;
          }
        }
        if(w>=479){ //Buffer2 꽉차면
        conv2spect=1;
        w=0; //스펙토그램 전환하라는 신호
        }
        }
      PT_YIELD(pt);
    }
  PT_END(pt);
  }

pt ptSpecto;
int spectoThread(struct pt* pt){
  PT_BEGIN(pt);
  for(;;){
    if(conv2spect==1){
      static size_t num_samples_read;
      TfLiteStatus yes_status = GenerateMicroFeatures(
      error_reporter,Buffer2, g_yes_30ms_sample_data_size,
      g_yes_feature_data_slice_size, yes_calculated_data, &num_samples_read);
      spectogramFile=SD.open("Specto.txt", FILE_WRITE);
      for(int i=0; i<g_yes_feature_data_slice_size; i++){
        spectogramFile.println(yes_calculated_data[i]);
        }
        Serial.println("InputSpectogrammmmmmmmmmmmmmmmmmmmmmmmm,,");
      spectogramFile.close();
      conv2spect=0;
      }
          PT_YIELD(pt);
      }
    PT_END(pt);
  }

pt ptSVWC;
int SVWCThread(struct pt* pt){
  PT_BEGIN(pt);
  for(;;){
    if(startSVWC==1){
      Serial.println("Start SVWC");
      
      spectogramFile=SD.open("Specto.txt");
      
      while(spectogramFile.available()){
        spec[w2]=spectogramFile.read();
      //Serial.println(spec[w2]);
      w2++;
      if(w2==spec_len-1){
        Serial.println("Something Counting...");
       static int* SV_input1 = spec;
      static float SV_output1[50];
      SV_call(SV_input1, SV_output1);
      Serial.println("111Called");
      normalize(SV_output1);
      Serial.println("normalized output");
      for(int i=0;i<50;i++){Serial.println(SV_output1[i]);}
      score1 = cos_sim(enroll_dvec, SV_output1);

      // 끝에서 SV돌리고 cosine similarity 계산
      static int* SV_input2 = spec+spec_len-49;
      static float SV_output2[50];
      SV_call(SV_input2, SV_output2);
      normalize(SV_output2);
      score2 = cos_sim(enroll_dvec, SV_output2);
      Serial.println("score is ::::");
      Serial.println(score1);
      Serial.println(score2);
 
      // 처음과 끝 모두 등록 화자가 아니라면 패스
      if (score1>thres && score2>thres){
      // 스펙트로그램을 91*40으로 쪼개 점수 합산
      for (int i=0; i<spec_len/(91*40); i++) {
        static int WC_output[11];
        WC_call(spec, WC_output);
        w3=1;
        num_words += argmax(WC_output, 11)+1;
        last_control=millis();
         PT_YIELD(pt);
      }
      Serial.println("Something Counted");
      }
      
      w2=0;
      Serial.println("w2 is now zero.");
      }
      }
      w3=0;
      startSVWC=0;
      spectogramFile.close();
      Serial.println("SVWC end, remove file...");
      SD.remove("Specto.txt");
      Serial.println("Removed Specto.txt");
      }
    PT_YIELD(pt);
    }
  PT_END(pt);
  }

/* 이거 사실 필요없을지도
pt ptCountWords; //모드1에서 단어수 세는용도 pt
int countWordsThread(struct pt* pt){
  PT_BEGIN(pt);
  for(;;){
    if(mode==1){
      if(w3=1){
      num_words=num_words+total_words;
      total_words=0;
      }
      //밑은 테스트용. 나중에 한번에 몰아서 하는 식으로 하면 비슷하게 되긴 할 듯?
      /*if(chk_counted==1){
      num_words=num_words+tmp_count;
      chk_counted=0;
      }
      }
    PT_YIELD(pt);
  }
  PT_END(pt);
  }
  */


pt ptDisplayNum; //모드에 맞는 디스플레이용 pt
int displayNumThread(struct pt* pt){
  PT_BEGIN(pt);
  for(;;){
    PT_YIELD(pt);
    if(light==1){ //일단 불이 들어와있으면
    if(mode==1){onRecording(num_words);}
    else if(mode==0){onStop(num_words);} //모드 1, 0일때 해당하는 화면 디스플레이
    else{
      LearningVoice();
      mode=1;
      }} // 모드 2일때 학습화면 디스플레이 및, 학습해주는 함수 호출
      else if(light==0){lightOff();} // 불 안들어와있으면 꺼줌.
      
      if(millis()-last_control>100000){ //마지막 조작에서 10초 이상 지나면 -> 일단 100초로 해둠.
        light=0;
        lightOff(); //불 끄고
        PT_YIELD(pt);
        PT_WAIT_UNTIL(pt,light==1); //다시 조작할때까지(불켜질 때까지) 기다림.
        }
      PT_YIELD(pt);
  }
  PT_END(pt);
  }




void setup() {
    Serial.begin(9600);
    pinMode(2, INPUT_PULLUP);
    pinMode(4, INPUT_PULLUP); //조작 버튼 두 개 핀모드 설정

     SD.begin(10); //SD카드 시작, CS핀 10번

    PT_INIT(&ptButton1Time); //PT들 시작
    PT_INIT(&ptButton2Time);
    PT_INIT(&ptState);
    //PT_INIT(&ptCountWords);
    PT_INIT(&ptDisplayNum);
    PT_INIT(&ptRecording);
    PT_INIT(&ptSpecto);
    PT_INIT(&ptSVWC);
    
  // SSD1306_SWITCHCAPVCC = generate display voltage from 3.3V internally
  if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;); // Don't proceed, loop forever
  }
  // Clear the OLED buffer
  display.clearDisplay();
  display.display();

  PDM.onReceive(onPDMdata); //PDM(음성신호) 작동 확인
  if(!PDM.begin(1,16000)){
    Serial.println("Failed to start PDM!");
    while(1);
    }

    // 에러 리포터 빌드
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  InitializeMicroFeatures(error_reporter);

  // enroll_dvec 임의로 설정
  for (int i=0;i<dvec_dim;i++){ enroll_dvec[i]=1/dvec_dim; }
  for (int i=0;i<10;i++){enroll_dvec[i]=1;}
  normalize(enroll_dvec);

  SV_setup();
  WC_setup();
}

void loop() {
  PT_SCHEDULE(button1TimeThread(&ptButton1Time));
  PT_SCHEDULE(button2TimeThread(&ptButton2Time));
  PT_SCHEDULE(stateThread(&ptState));
//  PT_SCHEDULE(countWordsThread(&ptCountWords));
  PT_SCHEDULE(displayNumThread(&ptDisplayNum));
  PT_SCHEDULE(recordingThread(&ptRecording));
  PT_SCHEDULE(spectoThread(&ptSpecto));
  PT_SCHEDULE(SVWCThread(&ptSVWC));
}



//마이크 입력받는 함수
void onPDMdata(){
  int bytesAvailable=PDM.available();
  PDM.read(Buffer, bytesAvailable);
  Read=bytesAvailable/2;
  }




void SV_call(int* input, float* output){
  // 모델 인풋 입력
  for (int i=0; i<kFeatureElementCount; i++) {
    SV_model_input->data.int8[i] = input[i]; //feature_buffer[i];
  }
  
  // 모델 실행 전 시간 기록
  start_time = millis();
  
  // 모델 실행, 에러 체크
  TfLiteStatus SV_invoke_status = SV_interpreter->Invoke();
  if (SV_invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }

  // 모델 실행 후 시간 기록
  end_time = millis();

  // 모델 아웃풋 얻기, 에러 체크
  TfLiteTensor* SV_model_output = SV_interpreter->output(0);
  if ((SV_model_output->dims->size != 2) || (SV_model_output->dims->data[0] != 1) || (SV_model_output->dims->data[1] != 50)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad output tensor parameters in model");
    return;
  }

  // 모델 아웃풋 출력
  for (int i=0; i<11; i++) {
    output[i] = (float) SV_model_output->data.int8[i];
  }
}




void WC_call(int* input, int* output){
  // 모델 인풋 입력
  for (int i=0; i<spec_len; i++) {
    WC_model_input->data.int8[i] = input[i]; //feature_buffer[i];
  }
  
  // 모델 실행 전 시간 기록
  start_time = millis();
  
  // 모델 실행, 에러 체크
  TfLiteStatus WC_invoke_status = WC_interpreter->Invoke();
  if (WC_invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "WC Invoke failed");
    return;
  }

  // 모델 실행 후 시간 기록
  end_time = millis();

  // 모델 아웃풋 얻기, 에러 체크
  TfLiteTensor* WC_model_output = WC_interpreter->output(0);
  if ((WC_model_output->dims->size != 2) || (WC_model_output->dims->data[0] != 1) || (WC_model_output->dims->data[1] != 11)) {
    TF_LITE_REPORT_ERROR(error_reporter, "WC Bad output tensor parameters in model");
    return;
  }

  // 모델 아웃풋 출력
  for (int i=0; i<11; i++) {
    output[i] = WC_model_output->data.int8[i];
  }
}



void SV_setup(){
    // 모델 로드, 버전 확인
  SV_model = tflite::GetModel(SV_model_data);
  if (SV_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model provided is schema version %d not equal to supported version %d.", SV_model->version(), TFLITE_SCHEMA_VERSION);
  }

  // 인터프리터 빌드  
  static tflite::MicroInterpreter SV_static_interpreter(
      SV_model, resolver, SV_tensor_arena, SV_tensorArenaSize, error_reporter);
  SV_interpreter = &SV_static_interpreter;

  // 메모리 할당, 에러 체크
  TfLiteStatus SV_allocate_status = SV_interpreter->AllocateTensors();
  if (SV_allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "SV AllocateTensors() failed");
  }

  // 모델 인풋 주소 얻기, 에러 체크
  SV_model_input = SV_interpreter->input(0);
  SV_model_input_buffer = SV_model_input->data.int8;
  if ((SV_model_input->dims->size!=3) || (SV_model_input->dims->data[0]!=1) || (SV_model_input->dims->data[1]!=49) || (SV_model_input->dims->data[2]!=40) || (SV_model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "SV Bad input tensor parameters in model");
  }
}


void WC_setup(){
  // 모델 로드, 버전 확인
  WC_model = tflite::GetModel(WC_model_data);
  if (WC_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, "WC Model provided is schema version %d not equal to supported version %d.", WC_model->version(), TFLITE_SCHEMA_VERSION);
  }

  // 인터프리터 빌드  
  static tflite::MicroInterpreter WC_static_interpreter(
      WC_model, resolver, WC_tensor_arena, WC_tensorArenaSize, error_reporter);
  WC_interpreter = &WC_static_interpreter;

  // 메모리 할당, 에러 체크
  TfLiteStatus WC_allocate_status = WC_interpreter->AllocateTensors();
  if (WC_allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "WC AllocateTensors() failed");
  }

  // 모델 인풋 주소 얻기, 에러 체크
  WC_model_input = WC_interpreter->input(0);
  WC_model_input_buffer = WC_model_input->data.int8;
  if ((WC_model_input->dims->size!=3) || (WC_model_input->dims->data[0]!=1) || (WC_model_input->dims->data[1]!=91) || (WC_model_input->dims->data[2]!=40) || (WC_model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "WC Bad input tensor parameters in model");
  }
}


void normalize(float* x){
  float sum=0;
  for (int i;i<dvec_dim;i++){ sum += x[i]*x[i]; }
  sum=sqrt(sum);
  for (int i;i<dvec_dim;i++){ x[i] /= sum; }
}

float cos_sim(float* x, float* y){
  float ret = 0;
  for (int i;i<dvec_dim;i++) { ret += x[i]*y[i]; }
  return ret;
}

int argmax(int* x, int len){
  int max_x = -127;
  int max_idx = 0;
  for (int i=0;i<len;i++){
    if (max_x<x[i]) {
      max_x = x[i];
      max_idx = i;
    }
  }
  return max_idx;
}
