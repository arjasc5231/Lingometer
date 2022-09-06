#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 32 // OLED display height, in pixels

// Declaration for an SSD1306 display connected to I2C (SDA, SCL pins)
// The pins for I2C are defined by the Wire-library. 
// On an arduino UNO:       A4(SDA), A5(SCL)
// On an arduino MEGA 2560: 20(SDA), 21(SCL)
// On an arduino LEONARDO:   2(SDA),  3(SCL), ...
#define OLED_RESET     -1 // Reset pin # (or -1 if sharing Arduino reset pin)
#define SCREEN_ADDRESS 0x3C ///< See datasheet for Address; 0x3D for 128x64, 0x3C for 128x32
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);


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
