#include "OLED.h"

volatile int mode=1;
// 1: 측정중 0: 측정 중지 2: 학습
volatile int num_words=100;

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
    num_words=num_words+50;
    }
  else if(mode==0){}
  else{}

}
