#include "Const.h" //기본 상수값 필요한 게 생기면 여기에.

#include "OLED.h" //OLED 출력 조작 관련 함수
#include "Mic.h" //음성 입력 관련 함수

#include "CountingWords.h" //단어 세는 함수
#include "LearningVoice.h" //목소리 학습 함수

volatile int mode=1;
// 1: 측정중 0: 측정 중지 2: 학습
volatile int light=1;
// 1: 화면 켜짐  0: 화면 꺼짐

volatile int button1_chk=1; //버튼1 조작용 변수
volatile unsigned long b1_in_time=1;
volatile unsigned long b1_out_time=1;

volatile unsigned int num_words=100; // 측정된 단어 수


void setup() {
    Serial.begin(9600);
    pinMode(2, INPUT_PULLUP);
    pinMode(4, INPUT_PULLUP); 
    
  // SSD1306_SWITCHCAPVCC = generate display voltage from 3.3V internally
  if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;); // Don't proceed, loop forever
  }
  // Clear the OLED buffer
  display.clearDisplay();
  display.display();

}

void loop() {
  //버튼 조작
  int button1=digitalRead(2);
  int button2=digitalRead(4);
  if(button1==0){
    if(button1_chk==1){
      button1_chk=0;
      b1_in_time=millis();
      }
    }
    else{
      if(button1_chk==0){
        b1_out_time=millis();
        unsigned long duration = b1_out_time - b1_in_time;
        if(duration<2000){
          if(mode==1){mode=0;}
          else if(mode==0){mode=1;}
          }
          else{
            num_words=0;
            }
         button1_chk=1;
        }
      }

   if(button2==0){
    if(light==1){light=0;}
    else if(light==0){light=1;}    
    }

  if(mode==1){
    onRecording(num_words);
//아래 주석은 딜레이때문에 버튼 테스트가 어려워서 일단 주석처리. 
//    delay(2000);
//    num_words=num_words+Count();
    }
  else if(mode==0){
    onStop(num_words);
    }
  else{
    LearningVoice();
    mode=1;    
    }

}
