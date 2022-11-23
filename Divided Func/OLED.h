#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

//단어 카운팅 중일 때 디스플레이
void onRecording(int num){
  display.clearDisplay();
  //숫자 디스플레이
  display.setTextSize(2);
  display.setTextColor(WHITE);
  display.setCursor(50,10);
  display.println(num);
  display.display();
  
  }

//측정 일시정지 중일 때 디스플레이
  void onStop(int num){
  display.clearDisplay();
  //숫자 디스플레이
  display.setTextSize(2);
  display.setTextColor(WHITE);
  display.setCursor(50,10);
  display.println(num);

  display.setTextSize(1);
  display.setCursor(10,5);
  display.println("STOP");
  
  display.display();
  }

  void onCounting(int num){
  display.clearDisplay();
  //숫자 디스플레이
  display.setTextSize(2);
  display.setTextColor(WHITE);
  display.setCursor(80,10);
  display.println(num);

  display.setTextSize(1);
  display.setCursor(5,5);
  display.println("Counting");
  
  display.display();
  }

//학습 모드일 때, 문장 출력하도록
  void forLearning(String a){
    display.clearDisplay();
    display.setTextSize(1.5);
    display.setTextColor(WHITE);
    display.setCursor(15,14);
    display.println(a);
    display.display();
    }

  void onSVing(String a){
    display.clearDisplay();
    display.setTextSize(3);
    display.setTextColor(WHITE);
    display.setCursor(15,14);
    display.println(a);
    display.display();
    }

  void lightOff(void){
    display.clearDisplay();
    display.display();
    }
