/*
  Active Learning Labs
  Harvard University 
  tinyMLx - Built-in Microphone Test
*/

#include <PDM.h>
#include <TinyMLShield.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"


// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;


// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 20000;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* GESTURES[] = {
  "hello",
  "goodbye"
};

#define NUM_GESTURES 2;


// PDM buffer
int32_t sampleBuffer[256];
volatile int samplesRead;

bool record = false;
int samples2Record = 3000;
int samplesRecorded = 0;
bool commandRecv = false;

void setup() {

  Serial.begin(9600);
  while (!Serial);  

  // Initialize the TinyML Shield
  initializeShield();

  PDM.onReceive(onPDMdata);
  // Initialize PDM microphone in mono mode with 16 kHz sample rate
  if (!PDM.begin(1, 16000)) {
    Serial.println("Failed to start PDM");
    while (1);
  }

  Serial.println("Welcome to the microphone test for the built-in microphone on the Nano 33 BLE Sense\n");
  Serial.println("Use the on-shield button or send the command 'click' to start and stop an audio recording");
  Serial.println("Open the Serial Plotter to view the corresponding waveform");

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);



// The property "dims" tells us the tensor's shape. It has one element for
// each dimension. Our input is a 2D tensor containing 1 element, so "dims"
// should have size 2.
Serial.println( tflInputTensor->dims->size);
// The value of each element gives the length of the corresponding tensor.
// We should expect two single element tensors (one is contained within the
// other).
Serial.println(tflInputTensor->dims->data[0]);
Serial.println(tflInputTensor->dims->data[1]);
// The input is a 32 bit floating point value
Serial.println( tflInputTensor->type);
Serial.println(kTfLiteFloat32);


}

void loop() {
  // see if the button is pressed and turn off or on recording accordingly
  bool clicked = readShieldButton();
  if (clicked){
    record = !record;
  }
  

  // display the audio if applicable
  if (samplesRead) {
    // print samples to serial plotter
    if (record) {
        for (int i = 0; i < samplesRead; i++) {
        if(samplesRecorded < samples2Record) {
            constexpr int32_t value_div = static_cast<int32_t>((25.6f * 26.0f) + 0.5f);

            featureValue = (sampleBuffer[i] * 256) + (value_div / 2)) / value_div;

            featureValue -= 128;

            if (featureValue < -128){
              featureValue = -128;
            }

            if (featureValue > 127){
              featureValue = 127;
            }

              Serial.print(int8_t(featueValue));
              tflInputTensor->data.int8[i] = int8_t(featueValue);

              if (samplesRecorded<samples2Record-1)
              {
                Serial.print(", ");
              }
              else{
                Serial.println("");
              }
            
            samplesRecorded++;
            }
            else{
              record = false;
              samplesRecorded = 0;
                                          // Run inferencing
                            TfLiteStatus invokeStatus = tflInterpreter->Invoke();
                            if (invokeStatus != kTfLiteOk) {
                              Serial.println("Invoke failed!");
                              while (1);
                              return;
                            }
                           
                            // Loop through the output tensor values from the model
                            for (int i = 0; i < 2; i++) {
                              Serial.print(GESTURES[i]);
                              Serial.print(": ");
                              Serial.println(tflOutputTensor->data.uint8_t[i]);

                            }
                            Serial.println();
                            
              break;
            }
        }

    }
    // clear read count
    samplesRead = 0;
  }
}

void onPDMdata() {
  // query the number of bytes available
  int bytesAvailable = PDM.available();

  // read data into the sample buffer
  PDM.read(sampleBuffer, bytesAvailable);

  samplesRead = bytesAvailable / 2;
}
