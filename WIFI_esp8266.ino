#include <ESP8266WiFi.h>

const char* ssid = "kimjihoo";       // Wi-Fi SSID
const char* password = "2121830kim!!";  // Wi-Fi 비밀번호

WiFiServer server(80);

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  Serial.println("Connected to WiFi");
  Serial.println(WiFi.localIP());  // ESP8266의 IP 주소 출력
  server.begin();
  Serial.println("Server started");
}

void loop() {
  WiFiClient client = server.available();

  if (client) {
    Serial.println("Client connected");
    String request = client.readStringUntil('\r');
    client.flush();
    Serial.println("Request: " + request);

    if (request.indexOf("/front") != -1) {
      Serial.println("front");
    } else if (request.indexOf("/left++") != -1) {
      Serial.println("left++");
    } else if (request.indexOf("/left+") != -1) {
      Serial.println("left+");
    } else if (request.indexOf("/left") != -1) {
      Serial.println("left");
    } else if (request.indexOf("/right++") != -1) {
      Serial.println("right++");
    } else if (request.indexOf("/right+") != -1) {
      Serial.println("right+");
    } else if (request.indexOf("/right") != -1) {
      Serial.println("right");
    } else if (request.indexOf("/stop") != -1) {
      Serial.println("stop");
    }

    client.stop();
    Serial.println("Client disconnected");
  }
}

