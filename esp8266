#include <ESP8266WiFi.h>

const char* ssid = "kimjihoo0525";
const char* password = "2121830kim!!";

WiFiServer server(80);

void setup() {
  Serial.begin(115200);  // Serial communication with Arduino
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  Serial.println("Connected to WiFi");
  Serial.println(WiFi.localIP());
  server.begin();
  Serial.println("Server started");
}

void loop() {
  WiFiClient client = server.available();

  if (client) {
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
