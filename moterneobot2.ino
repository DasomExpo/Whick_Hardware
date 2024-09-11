#include <Servo.h>

Servo servoLeft;  // 왼쪽 서보 모터
Servo servoRight; // 오른쪽 서보 모터
String cmd = "";  // 명령어 저장 변수

void setup() {
  Serial.begin(115200); // ESP8266과 동일한 속도로 설정
  servoLeft.attach(13); // 서보 모터 핀 연결
  servoRight.attach(12);
  Serial.println("Arduino Ready");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      // 명령을 처리하는 함수를 호출
      processCommand(cmd);
      cmd = ""; // 명령어 초기화
    } else {
      cmd += c; // 명령어 추가
    }
  }
}

void processCommand(String command) {
  if (command.startsWith("left++")) {
    Serial.println("Turning left++");
    servoLeft.write(1520);   // 왼쪽으로 최대 회전
    servoRight.write(1300); // 오른쪽으로 최대 회전
  } else if (command.startsWith("left+")) {
    Serial.println("Turning left+");
    servoLeft.write(1550);  // 왼쪽으로 중간 회전
    servoRight.write(1300); // 오른쪽으로 중간 회전
  } else if (command.startsWith("left")) {
    Serial.println("Turning left");
    servoLeft.write(1500);   // 왼쪽으로 약간 회전
    servoRight.write(1300); // 오른쪽으로 약간 회전
  } else if (command.startsWith("right++")) {
    Serial.println("Turning right++");
    servoLeft.write(1700);  // 오른쪽으로 최대 회전
    servoRight.write(1480);   // 왼쪽으로 최대 회전
  } else if (command.startsWith("right+")) {
    Serial.println("Turning right+");
    servoLeft.write(1700);  // 오른쪽으로 중간 회전
    servoRight.write(1450);  // 왼쪽으로 중간 회전
  } else if (command.startsWith("right")) {
    Serial.println("Turning right");
    servoLeft.write(1700);  // 오른쪽으로 약간 회전
    servoRight.write(1430);  // 왼쪽으로 약간 회전
  } else if (command.startsWith("front")) {
    Serial.println("Moving forward");
    servoLeft.write(1520);   // 서보 모터를 정면으로
    servoRight.write(1480);  // 서보 모터를 정면으로
  } else if (command.startsWith("stop")) {
    Serial.println("Stopping");
    servoLeft.write(1500);   // 서보 모터 정지 (중립 위치)
    servoRight.write(1500);  // 서보 모터 정지 (중립 위치)
  } else {
    Serial.println("Unknown command: " + command);
  }
}


