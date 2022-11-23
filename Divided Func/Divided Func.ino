#include <PDM.h>
#include "protothreads.h" //PT 라이브러리
#include <SPI.h> //SD카드 라이브러리
#include <SD.h> //SD카드 라이브러리

#include "Const.h" //기본 상수값 필요한 게 생기면 여기에.
#include "OLED.h" //OLED 출력 조작 관련 함수

// 텐서플로우 내장함수
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "speaker_verification_model_settings.h"
#include "speaker_verification_model.h"
#include "word_counting_model.h"
#include "feature_provider.h"


///////////////////////////////////////////////////////////////
////////////////////////////전역변수////////////////////////////
///////////////////////////////////////////////////////////////
// 에러 리포터 전역변수 선언
tflite::ErrorReporter* error_reporter = nullptr;

// feature provider
FeatureProvider* feature_provider = nullptr;

int total_words=0; // 총 단어 수
float enroll_dvec[dvec_dim]; // 화자 등록 d-vector (normalized)
float SV_thres = 0.0; // SV 역치
float VAD_thres = 0.0; // VAD 역치

// 스펙트로그램
int8_t spec[spec_dim]; // 실전에서는 (n*91)&40이 되도록 제로패딩

// SV 모델 전역변수 선언
const tflite::Model* SV_model = nullptr;
tflite::MicroInterpreter* SV_interpreter = nullptr;
TfLiteTensor* SV_model_input = nullptr;
constexpr int SV_tensorArenaSize = 22500; // 모델에 따라 크기 조정. 나중에 상수파일로 옮기기
uint8_t SV_tensor_arena[SV_tensorArenaSize];
int8_t* SV_model_input_buffer = nullptr;

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

// 모든 Op load. 필요한 Op만 로드해서 메모리를 줄일수도 있음
tflite::AllOpsResolver resolver;

// 음성 데이터 버퍼
const int audio_input=29515;
constexpr int Buffer_len =DEFAULT_PDM_BUFFER_SIZE*59;  //DEFAULT_PDM_BUFFER_SIZE*60, ((audio_input/DEFAULT_PDM_BUFFER_SIZE)+2)*DEFAULT_PDM_BUFFER_SIZE+1;
short Buffer[Buffer_len+(2*DEFAULT_PDM_BUFFER_SIZE)];
unsigned int Buffer_idx=0;


// mode=0: stop Recording,  mode=1: Recording,  mode2: Learning Voice
int mode = 1;
int light=1;
int SVWC = 0;
int spec_idx=0;
bool is_enroll=true;
bool voice=false;
bool chk_VAD=true;

//Files
File spectogramFile; //이 파일에 스펙토그램 넣을 예정임.
File enrollFile; //이 파일에 화자 목소리 저장 예정임.
File numFile; //이 파일에 단어 수 저장.

//조작
volatile int button1_chk=1; //버튼1 조작용 변수, 눌리면 0, 아니면 1
volatile int button2_chk=1; //버튼2 조작용 변수, 눌리면 0, 아니면 1
volatile unsigned long b1_in_time=1;
volatile unsigned long b1_out_time=1;
volatile unsigned long b2_in_time=1;
volatile unsigned long b2_out_time=1;
volatile unsigned long last_control=1;
volatile unsigned long now=1;

///////////////////////////////////////////////////////////////
/////////////////////////PT////////////////////////////////////
///////////////////////////////////////////////////////////////
pt ptState;
int stateThread(struct pt* pt){
  PT_BEGIN(pt);
  for(;;){
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
        if(mode==0){mode=1; Serial.println("now mode 1");}
        else if(mode==1){mode=0; Serial.println("now mode 0");} //짧게 누르면 측정 일시정지/ 재개하고
        } else {
          total_words=0; 
          SD.remove("Specto.txt");
          SD.remove("numW.txt");
          Serial.println("Reset TNW");
          } //길게 누르면 측정 초기화한다.
      PT_YIELD(pt);
    }
  PT_END(pt);
  }

pt ptButton2Time; // 버튼2 프레스 시간 확인 및 조작용 PT
int button2TimeThread(struct pt* pt){
  PT_BEGIN(pt);

  for(;;){
      PT_YIELD(pt);
      
      if(mode!=2){ //모드 0:SV, 모드1:WC, 모드2:학습
      PT_WAIT_UNTIL(pt, button2_chk==0);
      b2_in_time=millis();
      Serial.println("Button2 in");
      PT_YIELD(pt);
      
      PT_WAIT_UNTIL(pt,button2_chk==1);
      b2_out_time=millis();
      Serial.println("Button2 out");
      
      if(b2_out_time-b2_in_time<2000){
          if(mode==1){
            mode=0; 
            Serial.println("mode SV");
            total_words=0;
            }
          else if(mode==0){mode=1; Serial.println("mode WC");}
        }
        else{
          Serial.println("Start Voice Learning");
          mode=2;
          } //길게 누르면 불이 무조건 켜지면서 러닝 모드. 이후엔 다시 모드 0이나 1 상태로.
        }
        else{ //모드 2에 들어오면 학습하기
          PT_WAIT_UNTIL(pt, button2_chk==0);
          b2_in_time=millis();
          Serial.println("Start mode 2 Recording");
          for(int i=0; i<20000;i++){ //Buffer 비워주고
            Buffer[i+7000]=0;
            }
          Buffer_idx=0;
          PT_YIELD(pt);
          
          PT_WAIT_UNTIL(pt,button2_chk==1);
          b2_out_time=millis();
          Serial.println("mode 2 recording Finished");

          update_enroll_dvec(Buffer+8000,16000);
          normalize(enroll_dvec);
          Serial.println("enroll_dvec Updated"); //enroll_dvec 초기화
          
          SD.remove("enroll.txt");
          enrollFile=SD.open("enroll.txt", FILE_WRITE); //enroll_dvec 저장
          for(int i=0; i<dvec_dim; i++){ 
            Serial.print(enroll_dvec[i]); 
            enrollFile.println(enroll_dvec[i]);
            }
            enrollFile.close();
          Buffer_idx=0;
          mode=0;
          total_words=0;
           //학습 다했으면 SV로 만들기
          }
      PT_YIELD(pt);
    }
  PT_END(pt);
}


pt ptSpecto;
int spectoThread(struct pt* pt){
  PT_BEGIN(pt);
  for(;;){
    if (Buffer_idx >= Buffer_len){
      if(mode==1){
      chk_VAD=is_active(Buffer, audio_input);
      if(chk_VAD){
      feature_provider->PopulateFeatureData(error_reporter, Buffer, audio_input, spec);
      Serial.println("Spectogram Generated");
      SVWC=1;
      }}
      Buffer_idx=0;
      }
      PT_YIELD(pt);
    }
  PT_END(pt);
  }


pt ptSVWC;
int SVWCThread(struct pt* pt){
  PT_BEGIN(pt);
  for(;;){
    if (SVWC!=0){
    Serial.println("Start SVWC");
    if(mode==0){
      SV_process_audio();
      if(is_enroll){voice=true;}
      else{voice=false;}
    } else if(mode==1){
      WC_process_audio();
    }
    PT_YIELD(pt);
    SVWC=0;
    }
     PT_YIELD(pt);
    }
  PT_END(pt);
  }

pt ptDisplay; //모드에 맞는 디스플레이용 pt
int displayThread(struct pt* pt){
  PT_BEGIN(pt);
  for(;;){
    PT_YIELD(pt);
 //일단 불이 들어와있으면

    if(mode==1){onRecording(total_words);}
    else if(mode==0){if(voice){onSVing("YES");} else{onSVing("NO");}}
    else {forLearning("Press Button, and Say Something...");}//모드 0,1,2일때 해당하는 화면 디스플레이

      PT_YIELD(pt);
  }
  PT_END(pt);
  }



///////////////////////////////////////////////////////////////
///////////////////////////Func////////////////////////////////
///////////////////////////////////////////////////////////////



void setup() {
    Serial.begin(9600);
    pinMode(2, INPUT_PULLUP);
    pinMode(4, INPUT_PULLUP); //조작 버튼 두 개 핀모드 설정

    SD.begin(10); //SD카드 시작, CS핀 10번
    while(!SD.begin(10)){Serial.println("Initializing SD...");}
  
  // 에러 리포터 빌드
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // enroll_dvec 임의로 설정
  for (int i=0;i<10;i++){ enroll_dvec[i]=1; }

  if(SD.exists("enroll.txt")){ //기존 학습 결과 있으면 불러옴.
    enrollFile=SD.open("enroll.txt");
    int enroll_idx=0;
    while(enrollFile.available()){
      if(enroll_idx<dvec_dim){
        enroll_dvec[enroll_idx]=enrollFile.parseFloat();
        enroll_idx=enroll_idx+1;
      }else{enrollFile.read();}
      
    }
    enrollFile.close();
    }
  normalize(enroll_dvec);
 
 

  //OLED 기본 설정
    // SSD1306_SWITCHCAPVCC = generate display voltage from 3.3V internally
  if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;); // Don't proceed, loop forever
  }
  // Clear the OLED buffer
  display.clearDisplay();
  display.display();

  SV_setup();
  WC_setup();

  // FeatureProvider 생성
  static FeatureProvider static_feature_provider(kFeatureElementCount, spec);
  feature_provider = &static_feature_provider;

  // 마이크 설정
  PDM.onReceive(onPDMdata);
  if(!PDM.begin(1,16000)){
    Serial.println("Failed to start PDM!");
    while(1);
  }


  // PT 세팅
  PT_INIT(&ptState);
  PT_INIT(&ptSpecto);
  PT_INIT(&ptSVWC);
  PT_INIT(&ptButton2Time);
  PT_INIT(&ptDisplay);
  
}

void loop() {
  PT_SCHEDULE(stateThread(&ptState));
  PT_SCHEDULE(spectoThread(&ptSpecto));
  PT_SCHEDULE(SVWCThread(&ptSVWC));
  PT_SCHEDULE(button2TimeThread(&ptButton2Time));
  PT_SCHEDULE(displayThread(&ptDisplay));
}



///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////



void SV_call(int8_t* input, float* output){
  // 모델 인풋 입력
  for (int i=0; i<kFeatureElementCount; i++) {
    SV_model_input->data.int8[i] = input[i]; //feature_buffer[i];
  }
  
  // 모델 실행, 에러 체크
  TfLiteStatus SV_invoke_status = SV_interpreter->Invoke();
  if (SV_invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }

  // 모델 아웃풋 얻기, 에러 체크
  TfLiteTensor* SV_model_output = SV_interpreter->output(0);
  if ((SV_model_output->dims->size != 2) || (SV_model_output->dims->data[0] != 1) || (SV_model_output->dims->data[1] != 50)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad output tensor parameters in model");
    return;
  }

  // 모델 아웃풋 출력
  for (int i=0; i<dvec_dim; i++) {
    output[i] = (float) SV_model_output->data.int8[i];
  }
}



void WC_call(int8_t* input, int* output){ // output도 int8_t하는게 낫지 않을까
  // 모델 인풋 입력
  for (int i=0; i<spec_dim; i++) {
    WC_model_input->data.int8[i] = input[i]; //feature_buffer[i];
  }
  
  // 모델 실행, 에러 체크
  TfLiteStatus WC_invoke_status = WC_interpreter->Invoke();
  if (WC_invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "WC Invoke failed");
    return;
  }

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
  return (ret+1)/2;
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


// PDM callback function
void onPDMdata() {
  PDM.read(Buffer+Buffer_idx, DEFAULT_PDM_BUFFER_SIZE);
  if (Buffer_idx < Buffer_len) {Buffer_idx += DEFAULT_PDM_BUFFER_SIZE/2;}
}


void SV_process_audio() {
  int8_t* SV_input1 = spec;
  float SV_output1[50];
  SV_call(SV_input1, SV_output1);
  normalize(SV_output1);
  float score1 = cos_sim(enroll_dvec, SV_output1);

//  Serial.print("SV vector1: ");
//  for (int i=0; i<dvec_dim; i++){ Serial.print(SV_output1[i]); Serial.print(" ");}
  Serial.println();
  Serial.print("SV score1: "); Serial.println(score1);

  // 끝에서 SV돌리고 cosine similarity 계산
  int8_t* SV_input2 = spec+spec_dim-40*49;
  float SV_output2[50];
  SV_call(SV_input2, SV_output2);
  normalize(SV_output2);
  float score2 = cos_sim(enroll_dvec, SV_output2);

  Serial.println();
  Serial.print("SV score2: "); Serial.println(score2);

  // 처음과 끝 모두 등록 화자가 아니라면 패스
  is_enroll = score1>SV_thres && score2>SV_thres;
}


void WC_process_audio() {
  for (int i=0; i<spec_dim/(91*40); i++) {
    int WC_output[11];
    WC_call(spec, WC_output);
    total_words += argmax(WC_output, 11)+1;
  }
}


void update_enroll_dvec(short* audio, int audio_len){
  int8_t spec[kFeatureElementCount];
  size_t num_samples_read;
  feature_provider->PopulateFeatureData(error_reporter, audio, audio_len, spec);
  
  float SV_output[dvec_dim];
  SV_call(spec, SV_output);
  normalize(SV_output);
  for (int i=0;i<dvec_dim;i++){
    enroll_dvec[i] = SV_output[i]; 
  }
}

bool is_active(short* audio, int audio_len){
  unsigned long energy = 0;
  for(int i=0;i<audio_len;i++){ energy += audio[i]*audio[i];}
  if(energy>=audio_len*VAD_thres){return true;}
  else {return false;}
}
