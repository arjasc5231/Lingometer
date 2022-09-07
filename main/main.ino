#include<PDM.h>

#include "Const.h" //기본 상수값 필요한 게 생기면 여기에.

#include "OLED.h" //OLED 출력 조작 관련 함수

#include "CountingWords.h" //단어 세는 함수
#include "LearningVoice.h" //목소리 학습 함수

volatile int mode=1;
// 1: 측정중 0: 측정 중지 2: 학습
volatile int light=1;
// 1: 화면 켜짐  0: 화면 꺼짐

volatile int button1_chk=1; //버튼1 조작용 변수
volatile int button2_chk=1; //버튼2 조작용 변수
volatile unsigned long b1_in_time=1;
volatile unsigned long b1_out_time=1;
volatile unsigned long b2_in_time=1;
volatile unsigned long b2_out_time=1;
volatile unsigned long last_control=1;
volatile unsigned long now=1;

volatile unsigned int num_words=100; // 측정된 단어 수
short Buffer[256]; // 음성 신호 입력받을 변수
volatile int Read; //음성 신호 입력용 변수


void setup() {
    Serial.begin(9600);
    pinMode(2, INPUT_PULLUP);
    pinMode(4, INPUT_PULLUP); //조작 버튼 두 개 핀모드 설정
    
  // SSD1306_SWITCHCAPVCC = generate display voltage from 3.3V internally
  if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;); // Don't proceed, loop forever
  }
  // Clear the OLED buffer
  display.clearDisplay();
  display.display();

  PDM.onReceive(onPDMdata); //PDM 작동 확인
  if(!PDM.begin(1,16000)){
    Serial.println("Failed to start PDM!");
    while(1);
    }

}

void loop() {
  //버튼 조작
  int button1=digitalRead(2);
  int button2=digitalRead(4);
  if(button1==0){
    if(button1_chk==1){
      button1_chk=0;
      last_control=millis();
      b1_in_time=millis();
      }
    }
    else{
      if(button1_chk==0){
        b1_out_time=millis();
        unsigned long duration1 = b1_out_time - b1_in_time;
        if(duration1<2000){
          if(mode==1){mode=0;}
          else if(mode==0){mode=1;}
          }
          else{
            num_words=0;
            }
         button1_chk=1;
         last_control=millis();
         light=1;
        }
      }

   if(button2==0){
    if(button2_chk==1){
      button2_chk=0;
      last_control=millis();
      b2_in_time=millis();
      }
    }
    else{
      if(button2_chk==0){
      b2_out_time=millis();
      unsigned long duration2 = b2_out_time - b2_in_time;
      if(duration2<2000){
        if(light==1){
          light=0;
          lightOff();
          }
        else if(light==0){
          light=1;
          }      
        } 
       else{light=1; mode=2;}
      button2_chk=1;
      last_control=millis();
      }
    }

//마지막 조작 이후 10초가 지나면 화면 꺼짐
  now=millis();
  unsigned long after_control= now - last_control;
  if(after_control>10000){light=0; lightOff();}

//측정 상태에 따른 기능 변화
  if(mode==1){
    if(light==1){
    onRecording(num_words);
    delay(10);
    }
    num_words=num_words+Count();
    }
  else if(mode==0){
    if(light==1){
    onStop(num_words);
    }}
  else{
    if(light==1){
    LearningVoice();
    last_control=millis();
    mode=1;    
    }}

    //음성 신호 받는거 확인용.
    if(Read){
      for(int i=0; i<Read; i++){Serial.println(Buffer[i]);}
      Read=0;
      }

}

//마이크 입력받는 함수
void onPDMdata(){
  int bytesAvailable=PDM.available();
  PDM.read(Buffer, bytesAvailable);
  Read=bytesAvailable/2;
  }
