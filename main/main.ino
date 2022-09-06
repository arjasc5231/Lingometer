#include "OLED.h" //OLED 출력 조작 관련 함수
#include "CountingWords.h" //단어 세는 함수
#include "Const.h" //기본 상수값 필요한 게 생기면 여기에.

volatile int mode=1;
// 1: 측정중 0: 측정 중지 2: 학습
volatile int num_words=100; // 측정된 단어 수
String tmp_lr_words="Hello World"; //임시 학습용 문장

void setup() {
    Serial.begin(9600);
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
  // put your main code here, to run repeatedly:
  if(mode==1){
    onRecording(num_words);
    delay(2000);
    num_words=num_words+Count();
    }
  else if(mode==0){
    onStop(num_words);
    }
  else{
    forLearning(tmp_lr_words);
    }

}
