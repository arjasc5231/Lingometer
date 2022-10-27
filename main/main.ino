#include <PDM.h>
#include "protothreads.h" //PT 라이브러리

#include "Const.h" //기본 상수값 필요한 게 생기면 여기에.

#include "OLED.h" //OLED 출력 조작 관련 함수

#include "CountingWords.h" //단어 세는 함수
#include "LearningVoice.h" //목소리 학습 함수


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
volatile int Read; //음성 신호 입력용 변수

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
      Serial.println("button1ininininin");
      PT_YIELD(pt);
      
      PT_WAIT_UNTIL(pt,button1_chk==1);
      b1_out_time=millis();
      last_control=millis();
      light=1;
      
      if(b1_out_time-b1_in_time<2000){
        if(mode==0){mode=1;}
        else if(mode=1){mode=0;} //짧게 누르면 측정 일시정지/ 재개하고
        } else {num_words=0;} //길게 누르면 측정 초기화한다.
      Serial.println("button1outoutoutoutout");
      PT_YIELD(pt);
    }
  PT_END(pt)
  }

pt ptButton2Time; // 버튼2 프레스 시간 확인 및 조작용 PT
int button2TimeThread(struct pt* pt){
  PT_BEGIN(pt);

  for(;;){
      PT_YIELD(pt);
      
      PT_WAIT_UNTIL(pt, button2_chk==0);
      b2_in_time=millis();
      last_control=millis();
      Serial.println("button2ininininin");
      PT_YIELD(pt);
      
      PT_WAIT_UNTIL(pt,button2_chk==1);
      b2_out_time=millis();
      last_control=millis();
      
      if(b2_out_time-b2_in_time<2000){
        if(light==0){light=1;}
        else if(light==1){light=0;}
        } //짧게 누르면 불이 켜지거나 꺼지고
        else{
          mode=2;
          light=1;
          } //길게 누르면 불이 무조건 켜지면서 모드 2
      Serial.println("button2outoutoutoutout");
      PT_YIELD(pt);     
      
    }
  PT_END(pt)
  }


pt ptCountWords; //모드1에서 단어수 세는용도 pt
int countWordsThread(struct pt* pt){
  PT_BEGIN(pt);
  for(;;){
    if(mode==1){
      num_words=num_words+Count();
      }
    PT_YIELD(pt);
  }
  PT_END(pt);
  }


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
      
      if(millis()-last_control>10000){ //마지막 조작에서 10초 이상 지나면
        light=0;
        lightOff(); //불 끄고
        PT_YIELD(pt);
        PT_WAIT_UNTIL(pt,light==1); //다시 조작할때까지(불켜질 때까지) 기다림.
        }
      PT_YIELD(pt);
  }
  PT_END(pt);
  }


//마이크 입력받는 함수
void onPDMdata(){
  int bytesAvailable=PDM.available();
  PDM.read(Buffer, bytesAvailable);
  Read=bytesAvailable/2;
  }


void setup() {
    Serial.begin(9600);
    pinMode(2, INPUT_PULLUP);
    pinMode(4, INPUT_PULLUP); //조작 버튼 두 개 핀모드 설정

    PT_INIT(&ptButton1Time); //PT들 시작
    PT_INIT(&ptButton2Time);
    PT_INIT(&ptState);
    PT_INIT(&ptCountWords);
    PT_INIT(&ptDisplayNum);
    
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

}

void loop() {
  PT_SCHEDULE(button1TimeThread(&ptButton1Time));
  PT_SCHEDULE(button2TimeThread(&ptButton2Time));
  PT_SCHEDULE(stateThread(&ptState));
  PT_SCHEDULE(countWordsThread(&ptCountWords));
  PT_SCHEDULE(displayNumThread(&ptDisplayNum));


/*  //버튼 조작
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
*/
}
